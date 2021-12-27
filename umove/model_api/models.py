from django.db import models

class Video(models.Model):
    video_name = models.CharField(max_length=250)
    video = models.FileField(('Video'), upload_to='./static_cache/sent_vid/')

