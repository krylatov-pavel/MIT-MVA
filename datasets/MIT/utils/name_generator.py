import re

class NameGenerator(object):
    def __init__(self, file_extension):
        self.file_extension = file_extension

    def generate_name(self, index, rythm, record, start, end):
        template = "{}_{}_{}_{}-{}{}"
        return template.format(index, rythm, record, start, end, self.file_extension)

    def generate_aug_name(self, original, aug_name):
        return "{}_{}{}".format(
            original.rstrip(self.file_extension),
            aug_name,
            self.file_extension
        )

    def get_rythm(self, fname):
        regex = "^\d+_(?P<rythm>\(\w+)_(?P<record>\d+)_(?P<start>\d+)-(?P<end>\d+)"
        m = re.match(regex, fname)
        if m:
            return m.group('rythm')
        else:
            return None