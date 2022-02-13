from django.db import models


class SourceFile(models.Model):
    ''' Source file to predict '''
    file = models.FileField(upload_to='', null=True, verbose_name="")

    def __str__(self):
        ''' String presentation of file '''
        return str(self.file)

    def delete(self, *args, **kwargs):
        ''' Delete file from DB '''
        self.file.delete()
        super().delete(*args, **kwargs)


class PredictionFile(models.Model):
    ''' File with prediction '''
    file = models.FileField(upload_to='', null=True, verbose_name="")

    def __str__(self):
        ''' String presentation of file '''
        return str(self.file)

    def delete(self, *args, **kwargs):
        ''' Delete file from DB '''
        self.file.delete()
        super().delete(*args, **kwargs)