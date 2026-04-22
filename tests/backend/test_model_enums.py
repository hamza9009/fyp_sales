"""Regression tests for ORM enum mappings."""

from app.models.inventory_signals import AlertLevel, InventorySignal
from app.models.model_runs import ModelRun, ModelStatus


def test_model_run_status_enum_uses_database_labels():
    enum_type = ModelRun.__table__.c.status.type

    assert enum_type.enums == [status.value for status in ModelStatus]


def test_inventory_signal_alert_enum_uses_database_labels():
    enum_type = InventorySignal.__table__.c.alert_level.type

    assert enum_type.enums == [level.value for level in AlertLevel]
