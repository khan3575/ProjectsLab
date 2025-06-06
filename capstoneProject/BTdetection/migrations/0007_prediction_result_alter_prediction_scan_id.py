# Generated by Django 5.2 on 2025-05-30 09:21

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("BTdetection", "0006_remove_scan_upload_date_scan_scan_file"),
    ]

    operations = [
        migrations.AddField(
            model_name="prediction",
            name="result",
            field=models.CharField(default="Pending", max_length=20),
        ),
        migrations.AlterField(
            model_name="prediction",
            name="scan_id",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="prediction",
                to="BTdetection.scan",
            ),
        ),
    ]
