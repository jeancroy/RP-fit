from datetime import timedelta, datetime
from .hash import digest
from .files import save, try_load


class DataStore:

    def __init__(self, data=None, created_on=None, dependency_hash=None, max_age=None):
        self._data = data
        self._created_on = created_on if created_on is not None else datetime.now()
        self._dependency_hash = dependency_hash
        self._max_age = max_age

    def with_max_age(self, days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0):
        delta = timedelta(days, seconds, microseconds, milliseconds, minutes, hours, weeks)
        self._max_age = None if delta.total_seconds() == 0 else delta
        return self

    def with_dependency_on(self, *argv):
        self._dependency_hash = digest(*argv)
        return self

    def use_data(self, data):
        self._data = data
        return self

    def data(self):
        return self._data

    def is_valid(self):
        return self._data is not None

    def save_to(self, path):
        save(path, self)
        return self

    def try_read_and_validate(self, path):
        self._data = None

        conditional_data = try_load(path)
        if conditional_data is None or not isinstance(conditional_data, DataStore):
            return self
        if not conditional_data.validate_against(self._dependency_hash, self._max_age):
            return self

        self._data = conditional_data._data

        return self

    def validate_against(self, dependency_hash=None, max_age=None):

        if self._data is None:
            return False

        if dependency_hash is not None and self._dependency_hash != dependency_hash:
            return False

        if max_age is None:
            max_age = self._max_age

        if max_age is not None and self._created_on is not None:
            delta = datetime.now() - self._created_on
            if delta > max_age:
                return False

        return True



