# Generated by Django 4.2.3 on 2023-12-03 14:16

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("groupOneApp", "0011_alter_prediction_result"),
    ]

    operations = [
        migrations.AlterField(
            model_name="prediction",
            name="image",
            field=models.ImageField(blank=True, null=True, upload_to="images/"),
        ),
    ]