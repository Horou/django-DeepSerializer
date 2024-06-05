"""
A unique view-set for all your need of deep read and deep write, made easy
"""
from pprint import pprint

from django.db import connection
from django.db.models import Model
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.viewsets import ModelViewSet, ReadOnlyModelViewSet

from .serializers import DeepSerializer


###################################################################################################
#
###################################################################################################


class ReadOnlyDeepViewSet(ReadOnlyModelViewSet):
    """
    A read-only view-set that provides deep read functionality.
    This view-set is designed to make deep reading and writing easier.

    """
    _viewsets = {}
    use_case = "ReadOnly"

    def __init_subclass__(cls, **kwargs):
        """
        Initializes subclasses of ReadOnlyDeepViewSet.
        It saves important information such as all the view-sets inheriting this class
        and all possible fields for filtering or ordering the queryset.

        Args:
            kwargs: Additional keyword arguments.
        """
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'queryset') and cls.queryset is not None:
            model = cls.queryset.model
            cls._viewsets[cls.use_case + model.__name__ + "ViewSet"] = cls
            cls._possible_fields = cls.build_possible_fields(model, [])
            if cls.serializer_class is None:
                cls.serializer_class = DeepSerializer.get_serializer_class(model, use_case=cls.use_case)

    @classmethod
    def build_possible_fields(cls, parent_model: Model, excludes: list[Model]) -> set[str]:
        """
        Constructs a list of all possible fields for this view.
        This list is used to verify if a string can be used for filtering or ordering a queryset.

        Args:
            parent_model (Model): The parent model for which to build the relationships.
            excludes (list[Model]): The list of models to exclude.

        Returns:
            set[str]: A set of all possible fields for the view.
        """
        possible_fields = set()
        for field_relation in parent_model._meta.get_fields():
            if (model := field_relation.related_model) not in excludes:
                field_name = field_relation.name
                possible_fields.add(field_name)
                if model:
                    possible_fields.update((
                        f"{field_name}__{field}"
                        for field in cls.build_possible_fields(model, excludes + [parent_model])
                    ))
        return possible_fields

    @classmethod
    def init_router(cls, router, models: list) -> None:
        """
        Creates view-sets for all the models and registers them in the router.

        Args:
            router: A rest_framework router.
            models (list): A list of models to register in the router.
        """
        for model in models:
            router.register(model.__name__, cls.get_view_set_class(model), basename=model.__name__)

    def get_serializer(self, *args, **kwargs):
        """
        Returns the serializer instance that should be used for validating and deserializing input,
        and for serializing output.
        The depth of serialization and the relationships paths are determined based on the request parameters.

        Args:
            args: Additional positional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            Serializer: The serializer instance.
        """
        if self.request:
            params = self.request.query_params
            serializer_class = self.get_serializer_class()
            kwargs.setdefault('context', self.get_serializer_context())
            excludes = set(params.get("exclude", "").split(","))
            return serializer_class(
                *args,
                depth=int(params.get("depth", serializer_class.Meta.original_depth)),
                relations_paths={
                    path
                    for path in serializer_class._all_related_paths
                    if not any(path.startswith(exclude) for exclude in excludes if exclude)
                },
                **kwargs
            )
        else:
            return super().get_serializer(*args, **kwargs)

    def get_queryset(self):
        """
        Modifies the queryset based on the request parameters
        to filter, sort, and control the depth of the data returned.

        Filtering is done by specifying 'field_name=value' in the request parameters.
        Nested model fields can be filtered with 'field_name__field_name=value'.
        Sorting is specified with 'order_by=field_name'.
        If 'order_by' is a list, it will sort in the order of the list.
        The depth of the model display can be controlled with 'depth=10'.
        To exclude nested models, use 'exclude=foo,bar'.
        If the nested model to exclude is nested in another model, separate them with '__'.

        Returns:
            QuerySet: The modified queryset.
        """
        params = self.request.query_params
        serializer_class = self.get_serializer_class()
        excludes = set(params.get("exclude", "").split(","))
        queryset = serializer_class.optimize_queryset(
            self.queryset,
            {
                path
                for path in serializer_class._all_related_paths
                if not any(path.startswith(exclude) for exclude in excludes if exclude)
            },
            int(params.get("depth", serializer_class.Meta.original_depth))
        )
        if filter_by := {field: value for field, value in params.items() if field in self._possible_fields}:
            queryset = queryset.filter(**filter_by)
        if order_by := [field for field in params.get("order_by", "").split(",") if field in self._possible_fields]:
            queryset = queryset.order_by(*order_by)
        return queryset

    @classmethod
    def get_view_set_class(cls, model: Model, use_case: str = ""):
        """
        Retrieves or creates a view-set for the specified model and use case.
        Manually created view-sets inheriting DeepViewSet will automatically be used for their use case.

        If your view-set is only used in a specific use case, specify it in the use_case parameter.

        Args:
            model (Model): The model related to the desired view-set.
            use_case (str): The use case that this view-set will be used for.
                            If empty, it will be the main view-set for this model.

        Returns:
            ViewSet: The view-set for the specified model and use case.
        """
        view_set_name = use_case + model.__name__ + "ViewSet"
        if view_set_name not in cls._viewsets:
            _model, _use_case = model, use_case

            class CommonViewSet(cls):
                use_case = _use_case
                queryset = _model.objects

            CommonViewSet.__name__ = view_set_name
            CommonViewSet.__doc__ = f"""
            Generated ViewSet for the model: '{model.__name__}'
            Used for {use_case if use_case else 'Read and Write'}

            """ + ReadOnlyDeepViewSet.get_queryset.__doc__

        return cls._viewsets[view_set_name]


class DeepViewSet(ReadOnlyDeepViewSet, ModelViewSet):
    """
    A view-set that provides deep read and write functionality.
    This view-set is designed to make deep reading and writing easier.
    """
    use_case = ""


class DeepCreateViewSet(DeepViewSet):
    """
    A view-set that provides the action 'deep_create' for the deep_update_or_create function.
    This view-set is designed to have a ready-made view-set for creating or updating nested model.
    """
    @action(detail=False, methods=['post'])
    def deep_create(self, request):
        return Response(
            self.get_serializer().deep_update_or_create(self.queryset.model, request.data, raise_exception=True),
            status=status.HTTP_201_CREATED
        )

###################################################################################################
#
###################################################################################################
