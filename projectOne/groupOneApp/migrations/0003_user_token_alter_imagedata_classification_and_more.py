# Generated by Django 4.2.6 on 2023-11-22 23:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('groupOneApp', '0002_alter_prediction_image_delete_userimage'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='token',
            field=models.CharField(max_length=30, null=True),
        ),
        migrations.AlterField(
            model_name='imagedata',
            name='classification',
            field=models.IntegerField(choices=[(0, 'normal'), (1, 'benign'), (2, 'malignant')]),
        ),
        migrations.AlterField(
            model_name='imagedata',
            name='image',
            field=models.ImageField(upload_to=''),
        ),
        migrations.AlterField(
            model_name='imagedata',
            name='set',
            field=models.IntegerField(choices=[(0, 'test'), (1, 'train'), (2, 'validation')]),
        ),
        migrations.AlterField(
            model_name='prediction',
            name='image',
            field=models.ImageField(upload_to=''),
        ),
        migrations.AlterField(
            model_name='prediction',
            name='result',
            field=models.IntegerField(choices=[(0, 'low'), (1, 'medium'), (2, 'high')]),
        ),
    ]