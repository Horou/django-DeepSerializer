"""
A unique serializer for all your need of deep read and deep write, made easy
"""
import re
from collections import OrderedDict

from django.db.models import Model, Prefetch
from django.db.transaction import atomic
from rest_framework import serializers
from rest_framework.exceptions import ValidationError
from rest_framework.utils.field_mapping import (get_nested_relation_kwargs, )

########################################################################################################################
#
########################################################################################################################


PK_ERROR_MESSAGE = "Failed to Serialize"


class DeepSerializer(serializers.ModelSerializer):
    """
    A base serializer for handling deep serialization of Django models.

    This serializer inherits from the ModelSerializer and provides a framework for deep serialization of Django models.
    It includes methods for handling relationships between models and for initializing subclasses with specific metadata.

    Attributes:
        _serializers (dict): A dictionary mapping use cases and model names to serializers.
    """
    _serializers = {}

    def __init_subclass__(cls, **kwargs):
        """
        Initialize the subclass with specific metadata.

        This method is automatically called when the class is subclassed.
        It sets up the metadata for the subclass based on its Meta class.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "Meta"):
            if not hasattr(cls.Meta, "use_case"):
                cls.Meta.use_case = ""
            model = cls.Meta.model
            cls._serializers[f"{cls.Meta.use_case}{model.__name__}Serializer"] = cls
            excludes = [] if cls.Meta.fields == '__all__' else [
                field_relation.related_model
                for field_relation in model._meta.get_fields()
                if field_relation.related_model and field_relation.name not in cls.Meta.fields
            ]
            selects_related, prefetches_related, prefetches_with_selects = cls._build_related_paths(model, excludes)
            cls._selects_related, cls._prefetches_related = selects_related, prefetches_related
            cls._prefetches_related_with_selects = prefetches_with_selects
            cls._all_path_related = sorted(set(selects_related + prefetches_related + [
                f"{prefetch_path}__{select_path}"
                for prefetch_path, (_, prefetch_selects) in prefetches_with_selects.items()
                for select_path in prefetch_selects
            ]))
            forward_one, forward_many, reverse_one, reverse_many = cls._build_model_relationships(model, excludes)
            cls._forward_one_relationships, cls._forward_many_relationships = forward_one, forward_many
            cls._reverse_one_relationships, cls._reverse_many_relationships = reverse_one, reverse_many
            cls._all_relationships = {
                **forward_one,
                **forward_many,
                **{field_name: model for field_name, (model, _) in reverse_one.items()},
                **{field_name: model for field_name, (model, _) in reverse_many.items()}
            }
            cls.Meta.original_depth = cls.Meta.depth
            cls.Meta.read_only_fields = tuple({
                *(cls.Meta.read_only_fields if hasattr(cls.Meta, "read_only_fields") else []),
                *reverse_one,
                *reverse_many
            })

    def __init__(self, *args, **kwargs):
        """
        Initialize the serializer with specific arguments.

        This method is called when the serializer is instantiated.
        It sets up the depth and relations_paths attributes based on the provided arguments.

        Args:
            *args: Arbitrary positional arguments.
            **kwargs: Arbitrary keyword arguments.
        """
        self.Meta.depth = kwargs.pop("depth", self.Meta.original_depth)
        self.relations_paths = set(kwargs.pop("relations_paths", self.get_relationships_paths(depth=self.Meta.depth)))
        super().__init__(*args, **kwargs)

    @classmethod
    def _build_model_relationships(
            cls, parent_model: Model, excludes: list[Model]
    ) -> tuple[dict[str, Model], dict[str, Model], dict[str, tuple[Model, str]], dict[str, tuple[Model, str]]]:
        """
        Build relationships of a model.

        This method takes a parent model and a list of models to exclude,
        and returns a tuple of dictionaries mapping field names to models or tuples of models and reverse names for all relationships of the parent model that are not in the exclude list.

        Args:
            parent_model (Model): The parent model for which to build the relationships.
            excludes (list[Model]): The list of models to exclude.

        Returns:
            tuple: A tuple of four dictionaries:
            - forward_one_relations (dict[str, Model]): A dictionary mapping field names to models
                for all one-to-one and many-to-one relationships.
            - forward_many_relations (dict[str, Model]): A dictionary mapping field names to models
                for all many-to-many and one-to-many relationships.
            - reverse_one_relations (dict[str, tuple[Model, str]]): A dictionary mapping field names to tuples of models and reverse names for all one-to-one and many-to-one relationships that have a related name.
            - reverse_many_relations (dict[str, tuple[Model, str]]): A dictionary mapping field names to tuples of models and reverse names for all many-to-many and one-to-many relationships that have a related name.
        """
        forward_one_relations, forward_many_relations, reverse_one_relations, reverse_many_relations = {}, {}, {}, {}
        for field_relation in parent_model._meta.get_fields():
            if (model := field_relation.related_model) and model not in excludes:
                if field_relation.one_to_one or field_relation.many_to_one:
                    if not hasattr(field_relation, "field"):
                        forward_one_relations[field_relation.name] = model
                    elif field_relation.related_name:
                        reverse_one_relations[field_relation.name] = model, field_relation.field.name
                else:
                    if not hasattr(field_relation, "field"):
                        forward_many_relations[field_relation.name] = model
                    elif field_relation.related_name:
                        reverse_many_relations[field_relation.name] = model, field_relation.field.name
        return forward_one_relations, forward_many_relations, reverse_one_relations, reverse_many_relations

    @classmethod
    def _build_related_paths(
            cls, parent_model: Model, excludes: list[Model]
    ) -> tuple[list[str], list[str], dict[str, tuple[Model, list[str]]]]:
        """
        Generates paths for select_related and prefetch_related queries based on a model's relationships.

        Parameters:
            parent_model (Model): The model whose relationships are to be analyzed.
            excludes (list[Model]): Models to be omitted from the path generation.

        Returns:
            tuple: Contains three lists:
                - list[str]: Paths for select_related queries.
                - list[str]: Paths for prefetch_related queries.
                - dict[str, tuple[Model, list[str]]]: Mapping of prefetch_related paths to tuples of related model and its select_related paths.
        """
        selects_related, prefetches_related, prefetches_related_with_selects = [], [], {}
        for field_relation in parent_model._meta.get_fields():
            if (model := field_relation.related_model) and model not in excludes:
                if not hasattr(field_relation, "field") or field_relation.related_name:
                    selects, prefetches, prefetches_selects = cls._build_related_paths(model, excludes + [parent_model])
                    field_name = field_relation.name
                    if field_relation.one_to_one or field_relation.many_to_one:
                        selects_related += [field_name] + [f"{field_name}__{path}" for path in selects]
                        prefetches_related += [f"{field_name}__{path}" for path in prefetches]
                    else:
                        prefetches_related += [field_name] + [f"{field_name}__{path}" for path in prefetches]
                        if selects:
                            prefetches_related_with_selects[field_name] = model, selects
                    for path, prefetch_model_and_selects in prefetches_selects.items():
                        prefetches_related_with_selects[f"{field_name}__{path}"] = prefetch_model_and_selects
        return selects_related, prefetches_related, prefetches_related_with_selects

    @classmethod
    def optimize_queryset(cls, queryset, relations_paths: set[str], depth: int):
        """
        Optimize a queryset by selecting related paths and prefetching related objects.

        This method optimizes queryset by selecting related paths and prefetching related objects based on the serializer's metadata and the queryset relationships.
        It builds a list of select paths and a list of prefetch paths,
        and then applies these to the queryset using the select_related and prefetch_related methods.

        Args:
            queryset: The queryset to be optimized.
            relations_paths (set[str]): The set of relationship paths to consider for optimization.
            depth (int): The maximum length of the paths.

        Returns:
            The optimized queryset with related paths selected and related objects prefetched.
        """
        if depth and (selects := [
            select_path
            for select_path in cls._selects_related
            if select_path in relations_paths and depth > len(re.findall("__", select_path))
        ]):
            queryset = queryset.select_related(*selects)
        if prefetches := OrderedDict(
            (prefetch_path, prefetch_path)
            for prefetch_path in cls._prefetches_related
            if prefetch_path in relations_paths and depth >= len(re.findall("__", prefetch_path))
        ):
            for prefetch_path, (model, prefetch_selects) in cls._prefetches_related_with_selects.items():
                if prefetch_path in prefetches and (new_prefetch_selects := [
                    select_path
                    for select_path in prefetch_selects
                    if f"{prefetch_path}__{select_path}" in relations_paths
                    and depth > len(re.findall("__", f"{prefetch_path}__{select_path}"))
                ]):
                    prefetches[prefetch_path] = Prefetch(
                        prefetch_path,
                        queryset=model.objects.select_related(*new_prefetch_selects)
                    )
            queryset = queryset.prefetch_related(*prefetches.values())
        return queryset

    @classmethod
    def get_relationships_paths(cls, excludes: list[str] = [], depth: int = 0) -> list[str]:
        """
        Get a list of relations paths.

        This method takes a list of paths to exclude and a depth,
        and returns a list of prefetch_related paths that are not in the exclude list and have a length less than the depth plus 2.

        Args:
            excludes (list[str], optional): The list of paths to exclude. Defaults to an empty list.
            depth (int, optional): The maximum length of the paths. Defaults to 0.

        Returns:
            list[str]: A list of relations paths.
        """
        return [
            path
            for path in cls._all_path_related
            if len(re.findall("__", path)) <= depth and not any(
                path.startswith(exclude) for exclude in excludes if exclude
            )
        ]

    def get_default_field_names(self, declared_fields, model_info) -> list[str]:
        """
        Get a list of default field names.

        This method takes a list of declared fields and a model info object,
        and returns a list of default field names based on:
         - the primary key name.
         - the declared fields.
         - the fields of the model info.
         - the fields name present in relations_paths for this model.

        Args:
            declared_fields (list): The list of declared fields.
            model_info (ModelInfo): The model info object.

        Returns:
            list[str]: A list of default field names.
        """
        return (
                [model_info.pk.name] +
                list(declared_fields) +
                list(model_info.fields) +
                list(field for field in self.relations_paths if '__' not in field)
        )

    def build_nested_field(self, field_name: str, relation_info, nested_depth: int) -> tuple:
        """
        Build a nested field.

        This method takes a field name, a relation info object, and a nested depth,
        and returns a tuple of a serializer and a dictionary of nested relation kwargs for the nested field.

        Args:
            field_name (str): The name of the field.
            relation_info (RelationInfo): The relation info object.
            nested_depth (int): The nested depth.

        Returns:
            tuple: Tuple of a serializer and a dictionary of nested relation kwargs.
        """
        serializer = self.get_serializer_class(relation_info.related_model, use_case="Deep")
        nested_relation_kwargs = get_nested_relation_kwargs(relation_info)
        nested_relation_kwargs["depth"] = nested_depth - 1
        nested_relation_kwargs["relations_paths"] = set(
            path[len(field_name) + 2:]
            for path in self.relations_paths
            if path.startswith(f"{field_name}__")
        )
        return serializer, nested_relation_kwargs

    def _bulk_update_or_create(
            self, datas_and_nesteds: list[tuple[dict, dict]]
    ) -> list[tuple[dict, dict, any, OrderedDict]]:
        """
        Create or update multiple instances based on the data in datas_and_nesteds.

        This method takes a list of data and nested tuples, and creates or updates instances based on these inputs.
        The instances are updated or created based on the model's primary key.
        If the primary key exists, it will update the instance once and reuse this instance result when the primary key is found inside datas_and_nesteds again.
        If the primary key does not exist, it will create a new instance and reuse this instance result when the primary key is found inside datas_and_nesteds again.
        If there is no primary key, it will create a new instance without reusing others.

        Args:
            datas_and_nesteds (list[tuple[dict, dict]]): A list of tuples containing the data to be created or updated
                and the nested model representations to update the data representation with.

        Returns:
            list[tuple[dict, dict, any, dict]]: A list of tuples containing the primary_key or an error message
                and the representation or ERROR information, this for each tuple in datas_and_nesteds.
        """
        pk_name = self.Meta.model._meta.pk.name
        relations_paths = set(path for path in self.relations_paths if "__" not in path)
        instances = self.optimize_queryset(self.Meta.model.objects, relations_paths, 0).in_bulk(
            set(data[pk_name] for data, _ in datas_and_nesteds if pk_name in data)
        )
        serializer_class = self.get_serializer_class(self.Meta.model, "DeepCreate")
        processed_datas, created = [], {}
        for data, nested in datas_and_nesteds:
            found_pk = data.get(pk_name, None)
            if found_pk not in created:
                instance = instances.get(found_pk, None)
                serializer = serializer_class(
                    instance=instance,
                    data=data,
                    partial=bool(instance),
                    context=self.context,
                    depth=0,
                    relations_paths=relations_paths
                )
                if serializer.is_valid():
                    pk, representation = serializer.save().pk, serializer.data
                else:
                    pk, representation = PK_ERROR_MESSAGE, serializer.errors
                found_pk = found_pk if found_pk is not None else pk
                created[found_pk] = data, nested, pk, OrderedDict(representation)
            processed_datas.append(created[found_pk])
        return processed_datas

    def _clean_datas(
            self,
            datas: list[any],
            processed_datas: list[tuple[dict, dict, any, OrderedDict]],
            delete_models: list[Model]
    ) -> list[tuple[any, OrderedDict]]:
        """
        Clean the data representations.

        This method takes a list of data representations and a list of models to delete,
        and cleans the data representations based on these inputs.
        It will first check if the nested objects of a representation have fail and write an ERROR message if that is the case.
        It will then get the list of previous nested objects and delete them if they are part of the model in delete_models.

        Args:
            datas (list[any]): The list of data to process.
            processed_datas (list[tuple]): The list of data.
            delete_models (list[Model]): The list of models to delete.

        Returns:
            list[tuple[any, dict]]: A list of tuples containing the primary_key or an error message
                and the representation or ERROR information, this for each tuple in datas_and_nesteds.
        """
        pks_and_representations = []
        processed_datas_index = 0
        to_deletes: dict[str, tuple[Model, set]] = {}
        for data in datas:
            primary_key = representation = data
            if isinstance(representation, dict):
                data, nested, primary_key, representation = processed_datas[processed_datas_index]
                processed_datas_index += 1
                for field_name, model in self._all_relationships.items():
                    if model in delete_models and field_name in nested:
                        pk_name = model._meta.pk.name
                        old_primary_keys = representation[field_name]
                        new_primary_keys = nested[field_name]
                        if not isinstance(old_primary_keys, list):
                            old_primary_keys = [old_primary_keys]
                            new_primary_keys = [new_primary_keys]
                        if unused_primary_keys := set(
                            old_pk for old_pk in old_primary_keys if old_pk
                        ).difference(
                            pk[pk_name] if isinstance(pk, dict) else pk for pk in new_primary_keys if pk
                        ):
                            to_deletes.setdefault(field_name, (model, set()))
                            to_deletes[field_name][1].update(unused_primary_keys)
                if primary_key == PK_ERROR_MESSAGE:
                    representation["ERROR"] = PK_ERROR_MESSAGE
                    representation.move_to_end('ERROR', last=False)
                elif any(
                    (isinstance(field, list) and any(isinstance(item, dict) and "ERROR" in item for item in field))
                    or (isinstance(field, dict) and "ERROR" in field)
                    for field in nested.values()
                ):
                    representation["ERROR"] = f"{PK_ERROR_MESSAGE} nested objects"
                    representation.move_to_end('ERROR', last=False)
                representation.update(nested)
            pks_and_representations.append((primary_key, representation))
        for field_name, (model, primary_keys) in to_deletes.items():
            model.objects.filter(pk__in=primary_keys).delete()
        return pks_and_representations

    def _process_forward_one_relationships(self, datas_and_nesteds: list[tuple], delete_models: list[Model]):
        """
        Process the one_to_one and many_to_one relationship's models for the serializer model.

        This method takes a list of data and nested tuples and a list of models to delete.
        It regroups all nested data of one model and launches deep_process with the correct serializer for this model.
        It then updates the model fields in datas and nesteds with the primary_keys and representations returned by deep_process.

        Args:
            datas_and_nesteds (list[tuple]): The list of data and nested tuples.
            delete_models (list[Model]): The list of models to delete.
        """
        for field_name, model in self._forward_one_relationships.items():
            if field_name in self.relations_paths:
                filtered_datas_info, field_datas = [], []
                for data, nested in datas_and_nesteds:
                    field_data = data.get(field_name, None)
                    if field_data and isinstance(field_data, dict):
                        filtered_datas_info.append((data, nested))
                        field_datas.append(field_data)
                if filtered_datas_info:
                    results = self.fields[field_name].deep_process(field_datas, delete_models)
                    for (data, nested), result in zip(filtered_datas_info, results):
                        data[field_name], nested[field_name] = result

    def _process_forward_many_relationships(self, datas_and_nesteds: list[tuple], delete_models: list[Model]):
        """
        Process the many_to_many and one_to_many relationship's models for the serializer model.

        This method takes a list of data and nested tuples and a list of models to delete.
        It regroups all nested data of one model and launches deep_process with the correct serializer for this model.
        It then updates the model fields in datas and nesteds with the primary_keys and representations returned by deep_process.

        Args:
            datas_and_nesteds (list[tuple]): The list of data and nested tuples.
            delete_models (list[Model]): The list of models to delete.
        """
        for field_name, model in self._forward_many_relationships.items():
            if field_name in self.relations_paths:
                filtered_datas_info, field_datas = [], []
                for data, nested in datas_and_nesteds:
                    field_data = data.get(field_name, None)
                    if field_data and isinstance(field_data, list):
                        filtered_datas_info.append((data, nested, len(field_data)))
                        field_datas.extend(field_data)
                if filtered_datas_info:
                    results = self.fields[field_name].child.deep_process(field_datas, delete_models)
                    for data, nested, length in filtered_datas_info:
                        data[field_name], nested[field_name] = map(list, zip(*results[:length]))
                        results = results[length:]

    def _process_reverse_one_relationships(
            self, processed_datas: list[tuple[dict, dict, any, dict]], delete_models: list[Model]
    ):
        """
        Process the reverse one_to_one and many_to_one relationship's model for the serializer model.

        This method takes a list of data, a list of primary keys, a list of representations,
        and a list of models to delete.
        It regroups all nested data of one model,
        update the reverse field name with the primary_key of the parent and launches deep_process with the correct serializer for this model.
        It then replaces the dicts in representations with the representations returned by deep_process.

        Args:
            processed_datas (list[tuple]): The list of data.
            delete_models (list[Model]): The list of models to delete.
        """
        for field_name, (model, reverse_name) in self._reverse_one_relationships.items():
            if field_name in self.relations_paths:
                filtered_datas_info, datas = [], []
                for data, nested, primary_key, representation in processed_datas:
                    field_data = data.get(field_name, None)
                    if field_data and isinstance(field_data, dict):
                        field_data[reverse_name] = primary_key
                        filtered_datas_info.append((data, nested))
                        datas.append(field_data)
                if filtered_datas_info:
                    serializer = self.fields[field_name]
                    serializer.relations_paths.add(reverse_name)
                    results = serializer.deep_process(datas, delete_models)
                    for (data, nested), result in zip(filtered_datas_info, results):
                        data[field_name], nested[field_name] = result

    def _process_reverse_many_relationships(
            self, processed_datas: list[tuple[dict, dict, any, dict]], delete_models: list[Model]
    ):
        """
        Process the reverse many_to_many and one_to_many relationship's model for the serializer model.

        This method takes a list of data, a list of primary keys, a list of representations,
        and a list of models to delete.
        It regroups all nested data of one model, update the reverse field name with the primary_key of the parent
        and launches deep_process with the correct serializer for this model.
        It then replaces the dicts in representations with the representations returned by deep_process.

        Args:
            processed_datas (list[tuple]): The list of data.
            delete_models (list[Model]): The list of models to delete.
        """
        for field_name, (model, reverse_name) in self._reverse_many_relationships.items():
            if field_name in self.relations_paths:
                filtered_datas_info, datas = [], []
                for data, nested, primary_key, representation in processed_datas:
                    field_data = data.get(field_name, None)
                    if field_data and isinstance(field_data, list):
                        for item in field_data:
                            if isinstance(item, dict):
                                item[reverse_name] = primary_key
                        filtered_datas_info.append((data, nested, len(field_data)))
                        datas.extend(field_data)
                if filtered_datas_info:
                    serializer = self.fields[field_name].child
                    serializer.relations_paths.add(reverse_name)
                    results = serializer.deep_process(datas, delete_models)
                    for data, nested, length in filtered_datas_info:
                        data[field_name], nested[field_name] = map(list, zip(*results[:length]))
                        results = results[length:]

    def deep_process(self, datas: list[any], delete_models: list[Model]) -> list[tuple[any, any]]:
        """
        Deeply process a list of data.

        This method takes a list of data and a list of models to delete,
        and deeply processes the data based on these inputs.
        It processes forward and reverse relations, updates or creates data,
        and cleans the data representations.

        Args:
            datas (list[any]): The list of data to process.
            delete_models (list[Model]): The list of models to delete.

        Returns:
            list[tuple[any, any]]: A list of tuples, where each tuple contains a primary key and a data representation.
        """
        datas_and_nesteds = [(data, {}) for data in datas if isinstance(data, dict)]
        self._process_forward_one_relationships(datas_and_nesteds, delete_models)
        self._process_forward_many_relationships(datas_and_nesteds, delete_models)
        processed_datas = self._bulk_update_or_create(datas_and_nesteds)
        self._process_reverse_one_relationships(processed_datas, delete_models)
        self._process_reverse_many_relationships(processed_datas, delete_models)
        return self._clean_datas(datas, processed_datas, delete_models)

    def deep_update_or_create(
            self,
            model: Model,
            datas: list[dict] | dict,
            delete_models: list[Model] = [],
            verbose: bool = True,
            raise_exception: bool = False
    ) -> list[any]:
        """
        Create or update multiple instances with their nested instances at any depth based on the data in datas.

        This method takes a model, a list of data, a list of models to delete, and a verbosity flag.
        It then creates a list of models or a unique model with their nested models at any depth based on these inputs.
        If the resulting data is too big to be sent back,
        verbose=False is used to only send the primary key of the created model.
        If there have been errors, it will send the dictionary with the errors regardless of verbose.

        Args:
            model (Model): The model to be created or updated.
            datas (list[dict] | dict): The list of data to be created or updated.
            delete_models (list[Model], optional): The list of models to delete. Defaults to an empty list.
            verbose (bool, optional): The verbosity flag.
                If True, the method returns the full representation of the created models.
                If False, the method only returns the primary keys of the created models. Defaults to True.
            raise_exception (bool, optional): The raise_exception flag.
                If True, the method raise a ValidationError exception if an Error has been found during the processing.
                Defaults to False.

        Returns:
            list: A list of the created models or error info if there have been errors.
        """
        try:
            with atomic():
                primary_key, representation = zip(
                    *self.get_serializer_class(model, use_case="Deep")(
                        context=self.context,
                        depth=10
                    ).deep_process(
                        datas if isinstance(datas, list) else [datas],
                        delete_models
                    )
                )
                if any("ERROR" in data for data in representation if isinstance(data, dict)):
                    raise ValidationError(list(representation))
                return list(representation) if verbose else list(primary_key)
        except ValidationError as e:
            if raise_exception:
                raise e
            return e.detail

    @classmethod
    def get_serializer_class(cls, model: Model, use_case: str = ""):
        """
        Retrieve or create a serializer for the given model and its use case.

        This method takes a model and a use case, and returns a serializer for the model based on these inputs.
        If a manually created serializer that inherits from DeepViewSet exists for the use case,
        it will be return automatically.
        If the serializer is only used in a specific use case, the use case should be specified.

        If the serializer does not exist, a new one is created.
        The new serializer inherits from either the DeepSerializer or the main model serializer,
        and includes a Meta class with the model, a depth of 0, all fields, and the use case.

        Args:
            model (Model): The model of the serializer.
            use_case (str, optional): The use case for which the serializer will be used.
                If empty, the main serializer for this model will be used. Defaults to "".

        Returns:
            ModelSerializer: The serializer for the model and use case.
        """
        serializer_name = f"{use_case}{model.__name__}Serializer"
        if serializer_name not in cls._serializers:
            _model, _use_case = model, use_case

            class CommonSerializer(DeepSerializer):
                class Meta:
                    model = _model
                    depth = 0
                    fields = '__all__'
                    use_case = _use_case

            CommonSerializer.__name__ = serializer_name
            CommonSerializer.__doc__ = f'''
            A serializer for the model {_model}, used for {_use_case if _use_case else 'anything'}.

            This serializer inherits from the DeepSerializer and includes a Meta class with the model, a depth of 0,
            all fields, and the use case.
            The model and use case are provided when the serializer is created in the get_serializer_class method.
            '''

        return cls._serializers[serializer_name]

########################################################################################################################
#
########################################################################################################################
