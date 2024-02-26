from abc import abstractmethod
from abc import ABC

from typing import List

from fastapi import Request


class PermissionDependency(object):

    def __init__(self, permissions_classes: List):
        self.permissions_classes = permissions_classes

    def __call__(self, request: Request):
        for permissions_class in self.permissions_classes:
            permissions_class(request)


class BasePermission(ABC):

    def __init__(self, request: Request):
        self.request = request

    @abstractmethod
    def evaluate(self, request: Request):
        ...


class DefaultPermission(BasePermission):

    def evaluate(self, request: Request):
        pass
