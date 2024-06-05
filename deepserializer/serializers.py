"""
A unique serializer for all your need of deep read and deep write, made easy
"""
import re
from collections import OrderedDict

from django.db.models import Model, Prefetch, QuerySet
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
    A serializer for handling deep/nested serialization and deserialization of Django models.

    Attributes:
        _serializers (dict): A dictionary storing serializers for various models.
    """
    _serializers = {}

    def __init_subclass__(cls, **kwargs):
        """
        Initialize subclass by setting up model relationships and paths for related fields.

        This method sets up model relationships and paths for related fields by analyzing the model's metadata,
        identifying fields to exclude, and constructing related paths and relationships for deep serialization.

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
            related_paths = cls._build_related_paths(model, excludes)
            cls._selects_related, cls._prefetches_related, cls._selects_in_prefetches_related = related_paths
            cls._all_related_paths = sorted(set(related_paths[0] + related_paths[1] + [
                f"{prefetch_path}__{select_path}"
                for prefetch_path, (_, prefetch_selects) in related_paths[2].items()
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
        Initialize the DeepSerializer instance.

        This constructor initializes the DeepSerializer instance, setting the depth and relations paths
        based on provided arguments or default values.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.Meta.depth = kwargs.pop("depth", self.Meta.original_depth)
        self.relations_paths = {
            path
            for path in kwargs.pop("relations_paths", self._all_related_paths)
            if self.Meta.depth >= len(re.findall("__", path))
        }
        super().__init__(*args, **kwargs)

    @classmethod
    def _build_model_relationships(cls, parent_model: Model, excludes: list[Model]) -> tuple[
        dict[str, Model], dict[str, Model], dict[str, tuple[Model, str]], dict[str, tuple[Model, str]]
    ]:
        """
        Build relationships for the given model, excluding specified models.

        This method identifies and categorizes the forward and reverse relationships of the given model,
        excluding any specified models, and returns dictionaries of these relationships.

        Args:
            parent_model (Model): The parent model.
            excludes (list[Model]): List of models to exclude.

        Returns:
            tuple: Containing dictionaries for forward one-to-one, forward many-to-one, reverse one-to-many,
            and reverse many-to-many relationships.
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
    def _build_related_paths(cls, parent_model: Model, excludes: list[Model]) -> tuple[
        list[str], list[str], dict[str, tuple[Model, list[str]]]
    ]:
        """
        Build related paths for the given model, excluding specified models.

        This method recursively builds and categorizes the related paths for the given model, excluding any specified models.

        Args:
            parent_model (Model): The parent model.
            excludes (list[Model]): List of models to exclude.

        Returns:
            tuple: Containing lists of selects related, prefetches related, and selects in prefetches related.
        """
        selects_related, prefetches_related, selects_in_prefetches_related = [], [], {}
        for field_relation in parent_model._meta.get_fields():
            model, relation_name = field_relation.related_model, field_relation.name
            if model and model not in excludes and (
                    not hasattr(field_relation, "field") or field_relation.related_name
            ):
                selects, prefetches, prefetches_selects = cls._build_related_paths(model, excludes + [parent_model])
                if field_relation.one_to_many or field_relation.many_to_many:
                    if selects:
                        selects_in_prefetches_related[relation_name] = model, selects
                    prefetches_related.append(relation_name)
                else:
                    selects_related += [relation_name] + [f"{relation_name}__{path}" for path in selects]
                prefetches_related.extend(f"{relation_name}__{path}" for path in prefetches)
                for path, prefetch_model_and_selects in prefetches_selects.items():
                    selects_in_prefetches_related[f"{relation_name}__{path}"] = prefetch_model_and_selects
        return selects_related, prefetches_related, selects_in_prefetches_related

    @classmethod
    def optimize_queryset(cls, queryset: QuerySet, relations_paths: set[str], depth: int) -> QuerySet:
        """
        Optimize the queryset based on the given relationships paths and depth.

        This method optimizes the queryset by applying `select_related` and `prefetch_related` to reduce database queries,
        based on the specified relationship paths and depth.

        Args:
            queryset (QuerySet): The queryset to optimize.
            relations_paths (set[str]): The set of related paths to include.
            depth (int): The maximum depth for relationships.

        Returns:
            QuerySet: The optimized queryset.
        """
        relations_paths = {path for path in relations_paths if depth >= len(re.findall("__", path))}
        if selects := [path for path in cls._selects_related if path in relations_paths]:
            queryset = queryset.select_related(*selects)
        if prefetches := OrderedDict((path, path) for path in cls._prefetches_related if path in relations_paths):
            for prefetch_path, (model, prefetch_selects) in cls._selects_in_prefetches_related.items():
                if prefetch_path in prefetches and (new_prefetch_selects := [
                    path for path in prefetch_selects if f"{prefetch_path}__{path}" in relations_paths
                ]):
                    prefetches[prefetch_path] = Prefetch(
                        prefetch_path,
                        queryset=model.objects.select_related(*new_prefetch_selects)
                    )
            queryset = queryset.prefetch_related(*prefetches.values())
        return queryset

    def get_default_field_names(self, declared_fields, model_info) -> list[str]:
        """
        Get the default field names for the serializer.

        This method constructs the default field names by combining primary key, declared fields, model fields,
        and relationships paths up to a certain depth.

        Args:
            declared_fields (dict): The declared fields of the serializer.
            model_info (Model): The model being serialized.

        Returns:
            list[str]: List of default field names.
        """
        return (
                [model_info.pk.name] +
                list(declared_fields) +
                list(model_info.fields) +
                list(field for field in self.relations_paths if '__' not in field)
        )

    def build_nested_field(self, field_name: str, relation_info, nested_depth: int) -> tuple:
        """
        Build a nested field for the given field name and relationship info.

        This method constructs a nested field by determining the appropriate serializer and relationship kwargs
        for the given field and nested relationship info.

        Args:
            field_name (str): The name of the field.
            relation_info (RelationInfo): The relationship information.
            nested_depth (int): The depth of the nested relationship.

        Returns:
            tuple: Containing the serializer and nested relation kwargs.
        """
        nested_relation_kwargs = get_nested_relation_kwargs(relation_info)
        nested_relation_kwargs["depth"] = nested_depth - 1
        nested_relation_kwargs["relations_paths"] = {
            path[len(field_name) + 2:] for path in self.relations_paths if path.startswith(f"{field_name}__")
        }
        return self.get_serializer_class(relation_info.related_model, use_case="Deep"), nested_relation_kwargs

    def _bulk_update_or_create(
            self, datas_and_nesteds: list[tuple[dict, dict]]
    ) -> list[tuple[dict, dict, any, OrderedDict]]:
        """
        Bulk update or create instances based on the provided data.

        The function processes a list of data and nested relationship tuples. It first identifies primary keys (PKs)
        and fetches existing instances in bulk. For each data tuple, it determines if the instance already exists:
        - If it exists, it updates the instance.
        - If it doesn't exist, it creates a new instance.
        The function do not reprocess data with the same pk.

        Args:
            datas_and_nesteds (list[tuple[dict, dict]]): A list of tuples containing the data and nested relationships.

        Returns:
            list[tuple[dict, dict, any, OrderedDict]]: A list of tuples containing the data, nested relationships,
            primary key, and ordered dictionary representation of the instances.
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
        Clean and process the provided data, handling deletions as needed.

        The function iterates through the provided data and processed data, updating primary keys and handling deletions
        of unused nested relationships. It compiles a list of primary keys and their representations, and deletes any
        unused related models as specified.

        Args:
            datas (list[any]): List of input data.
            processed_datas (list[tuple[dict, dict, any, OrderedDict]]): Processed data with primary keys and representations.
            delete_models (list[Model]): List of models to delete unused nested relationships.

        Returns:
            list[tuple[any, OrderedDict]]: List of tuples containing primary keys and their ordered dictionary representations.
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
        Process forward many-to-one and one-to-one relationships for the provided data.

        The function iterates through the data, identifying and processing forward one-to-one relationships.
        It collects the relevant data, processes it using the related field's serializer, and updates the original data
        with the processed results.

        Args:
            datas_and_nesteds (list[tuple]): List of data and nested relationships.
            delete_models (list[Model]): List of models to delete unused nested relationships.
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
        Process forward many-to-many and one-to-many relationships for the provided data.

        The function iterates through the data, identifying and processing forward many-to-one relationships.
        It collects the relevant data, processes it using the related field's serializer, and updates the original data
        with the processed results.

        Args:
            datas_and_nesteds (list[tuple]): List of data and nested relationships.
            delete_models (list[Model]): List of models to delete unused nested relationships.
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
        Process reverse many-to-one and one-to-one relationships for the provided data.

        The function iterates through the processed data, identifying and processing reverse one-to-many relationships.
        It collects the relevant data, processes it using the related field's serializer, and updates the original data
        with the processed results.

        Args:
            processed_datas (list[tuple[dict, dict, any, dict]]): Processed data with primary keys and representations.
            delete_models (list[Model]): List of models to delete unused nested relationships.
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
        Process reverse many-to-many and one-to-many relationships for the provided data.

        The function iterates through the processed data, identifying and processing reverse many-to-many relationships.
        It collects the relevant data, processes it using the related field's serializer, and updates the original data
        with the processed results.

        Args:
            processed_datas (list[tuple[dict, dict, any, dict]]): Processed data with primary keys and representations.
            delete_models (list[Model]): List of models to delete unused nested relationships.
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

    def deep_process(self, datas: list, delete_models: list[Model]) -> list[tuple]:
        """
        Orchestrate the deep processing of data, handling relationships and deletions.

        The function processes the provided data by:
        - Processing forward one-to-one and many-to-one relationships.
        - Bulk updating or creating instances.
        - Processing reverse one-to-many and many-to-many relationships.
        - Cleaning the data and handling deletions.

        Args:
            datas (list): List of input data.
            delete_models (list[Model]): List of models to delete unused nested relationships.

        Returns:
            list[tuple]: List of tuples containing primary keys and their ordered dictionary representations.
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
    ) -> list:
        """
        Handle the deep update or creation of instances of the model using the provided data.

        The function processes the provided data, performs deep updates or creations of instances,
        and returns the representations of the processed data. It handles errors and can raise exceptions
        if specified.

        Args:
            model (Model): The model class to update or create instances for.
            datas (list[dict] | dict): The input data to process.
            delete_models (list[Model], optional): List of models to delete unused nested relationships. Defaults to [].
            verbose (bool, optional): Whether to return full representations or just primary keys. Defaults to True.
            raise_exception (bool, optional): Whether to raise exceptions on validation errors. Defaults to False.

        Returns:
            list: List of processed data representations or primary keys.
        """
        try:
            with atomic():
                primary_key, representation = zip(
                    *self.get_serializer_class(model, use_case="Deep")(context=self.context, depth=10).deep_process(
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
    def get_serializer_class(cls, model: Model, use_case: str = "") -> "DeepSerializer":
        """
        Retrieve the serializer class for a given model and use case.

        The function generates a serializer class dynamically if it doesn't already exist,
        based on the provided model and use case. The generated class includes a Meta class
        with the model, a depth of 0, and all fields.

        Args:
            model (Model): The model class to generate the serializer for.
            use_case (str, optional): The use case for the serializer. Defaults to "".

        Returns:
            DeepSerializer: The generated or cached serializer class.
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
