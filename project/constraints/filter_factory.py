class FilterFactory:
    """过滤器工厂
    """

    __FILTERS__ = {}

    @classmethod
    def add(cls, constraint_name, constraint):
        if constraint_name not in cls.__FILTERS__:
            cls.__FILTERS__[constraint_name] = constraint

    @classmethod
    def get(cls, constraint_name):
        if constraint_name not in cls.__FILTERS__:
            raise KeyError(f'No exist constraint with name `{constraint_name}`, you need register it at first!!!')
        return cls.__FILTERS__[constraint_name]

    @classmethod
    def get_all(cls):
        return list(cls.__FILTERS__.values())

    @classmethod
    def count(cls):
        return len(cls.__FILTERS__)

def register_filter(name):
    def inner(clazz):
        FilterFactory.add(name, clazz)
    return inner