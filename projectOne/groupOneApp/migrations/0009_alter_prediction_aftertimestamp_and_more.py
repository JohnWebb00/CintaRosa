# Generated by Django 4.2.3 on 2023-11-30 11:46

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("groupOneApp", "0008_remove_prediction_timestamp_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="prediction",
            name="afterTimestamp",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="prediction",
            name="beforeTimestamp",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
