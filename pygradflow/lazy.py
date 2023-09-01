from functools import update_wrapper


class lazyprop(property):
    def __init__(self, method, fget=None, fset=None, fdel=None, doc=None):
        self.method = method
        self.cache_name = "_cached_{}".format(self.method.__name__)

        doc = doc or method.__doc__
        super(lazyprop, self).__init__(fget=fget, fset=fset, fdel=fdel, doc=doc)

        update_wrapper(self, method)

    def __get__(self, instance, owner):
        if instance is None:
            return self

        try:
            return getattr(instance, self.cache_name)
        except AttributeError:
            if self.fget is not None:
                result = self.fget(instance)
            else:
                result = self.method(instance)

            setattr(instance, self.cache_name, result)

            return result
