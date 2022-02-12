from django.db import models

class PredictionFile(models.Model):
    file = models.FileField(upload_to='', null=True, verbose_name="")

    def __str__(self):
        return str(self.videofile)