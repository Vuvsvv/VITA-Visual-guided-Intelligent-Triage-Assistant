from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_department_list_uses_db_reference_rows():
    with patch(
        "app.routes.reference.fetch_reference_departments",
        return_value=[
            {"dept_id": "5", "parent_dept": "內科系", "child_dept": "一般內科"},
            {"dept_id": "7", "parent_dept": "外科系", "child_dept": "一般骨科"},
        ],
    ):
        response = client.get("/reference/departments")

    assert response.status_code == 200
    assert response.json() == {
        "departments": [
            {"dept_id": "5", "parentDept": "內科系", "childDept": "一般內科"},
            {"dept_id": "7", "parentDept": "外科系", "childDept": "一般骨科"},
        ]
    }


def test_general_orthopedics_returns_only_its_doctors():
    requested_departments = []

    def exact_department_doctors(department: str):
        requested_departments.append(department)
        return [
            {"doctor_id": "12", "name": "骨科醫師甲"},
            {"doctor_id": "13", "name": "骨科醫師乙"},
        ]

    with patch(
        "app.routes.reference.fetch_reference_doctors",
        side_effect=exact_department_doctors,
    ):
        response = client.get("/reference/doctors", params={"department": "一般骨科"})

    assert response.status_code == 200
    assert requested_departments == ["一般骨科"]
    assert response.json() == {
        "department": "一般骨科",
        "doctors": [
            {"doctor_id": "12", "name": "骨科醫師甲"},
            {"doctor_id": "13", "name": "骨科醫師乙"},
        ],
    }


def test_reference_api_failure_never_returns_fake_data():
    with patch(
        "app.routes.reference.fetch_reference_departments",
        side_effect=RuntimeError("db unavailable"),
    ):
        response = client.get("/reference/departments")

    assert response.status_code == 503
    assert response.json() == {"detail": "正式科別資料目前無法載入，請稍後重試。"}
    assert "departments" not in response.json()


def test_doctor_api_failure_never_returns_fake_data():
    with patch(
        "app.routes.reference.fetch_reference_doctors",
        side_effect=RuntimeError("db unavailable"),
    ):
        response = client.get("/reference/doctors", params={"department": "一般骨科"})

    assert response.status_code == 503
    assert response.json() == {"detail": "正式醫師資料目前無法載入，請稍後重試。"}
    assert "doctors" not in response.json()
