import logging
from datetime import date, datetime, timedelta
from typing import Any, List
from zoneinfo import ZoneInfo
import pyodbc

from .config import get_settings

logger = logging.getLogger(__name__)
CLOSED_SCHEDULE_STATUSES = {"請假休診", "額滿", "額滿關診"}
OPEN_SCHEDULE_STATUSES = {"open", "available", "可掛號"}
QUICK_SEARCH_VISIT_TYPE = "複診"
_last_db_fallback_reason = ""
TAIPEI_ZONE = ZoneInfo("Asia/Taipei")


class DatabaseUnavailableError(RuntimeError):
    """Canonical SQL data could not be read; this is not an empty-result condition."""


def create_db_connection():
    settings = get_settings()
    conn_parts = [
        f"DRIVER={{{settings.db_driver}}};",
        f"SERVER={settings.db_server};",
        f"DATABASE={settings.db_name};",
        f"UID={settings.db_user};",
        f"PWD={settings.db_password};",
        "Encrypt=yes;",
    ]
    if settings.db_trust_server_certificate or _is_local_sql_server(settings.db_server):
        conn_parts.append("TrustServerCertificate=yes;")
    conn_str = "".join(conn_parts)
    return pyodbc.connect(conn_str)


def _is_local_sql_server(server: str) -> bool:
    host = str(server or "").split(",", 1)[0].strip().lower()
    return host in {"127.0.0.1", "localhost", ".", "(local)"}


def _set_last_db_fallback_reason(reason: str) -> None:
    global _last_db_fallback_reason
    _last_db_fallback_reason = reason


def _schedule_status_is_available(status: Any) -> bool:
    text = str(status or "").strip()
    if not text:
        return True
    if text in CLOSED_SCHEDULE_STATUSES:
        return False
    return text.lower() in OPEN_SCHEDULE_STATUSES


def _merge_visit_type(slot: dict[str, Any], visit_type: str) -> None:
    visit_type = visit_type.strip()
    if not visit_type:
        return
    current = str(slot.get("visit_type") or "").strip()
    if not current:
        slot["visit_type"] = visit_type
        return
    values = [value for value in current.split("/") if value]
    if visit_type not in values:
        values.append(visit_type)
        slot["visit_type"] = "/".join(values)


def fetch_active_departments() -> List[dict]:
    conn = None
    try:
        conn = create_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                d.dept_id,
                dc.name AS parent_dept,
                d.name AS child_dept
            FROM Department d
            JOIN DepartmentCategory dc ON d.category_id = dc.category_id
            ORDER BY dc.name, d.name
        """)
        rows = cursor.fetchall()
        return [
            {"dept_id": int(row[0]), "parent_dept": row[1] or "", "child_dept": row[2] or ""}
            for row in rows
            if row[0] is not None and row[2]
        ]
    except DatabaseUnavailableError:
        raise
    except Exception as exc:
        logger.error("DB active departments unavailable error_type=%s", type(exc).__name__)
        raise DatabaseUnavailableError("正式科別資料目前無法載入。") from exc
    finally:
        if conn is not None:
            conn.close()


def fetch_reference_departments() -> List[dict]:
    """Return canonical department master data without any mock fallback."""
    conn = None
    try:
        conn = create_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                d.dept_id,
                dc.name AS parent_dept,
                d.name AS child_dept
            FROM Department d
            JOIN DepartmentCategory dc ON d.category_id = dc.category_id
            ORDER BY dc.name, d.name
        """)
        return [
            {
                "dept_id": str(row[0]),
                "parent_dept": str(row[1] or "").strip(),
                "child_dept": str(row[2] or "").strip(),
            }
            for row in cursor.fetchall()
            if str(row[2] or "").strip()
        ]
    except Exception as exc:
        raise DatabaseUnavailableError("正式科別資料目前無法載入。") from exc
    finally:
        if conn is not None:
            conn.close()


def fetch_reference_doctors(
    department_name: str,
    *,
    department_id: int | None = None,
) -> List[dict]:
    """Return canonical active doctors historically associated with one exact department.

    Doctor has no department foreign key in the current schema, so Schedule is used only
    as the association table. There is intentionally no date, status, or visit-type filter.
    """
    if department_id is not None and (
        isinstance(department_id, bool)
        or not isinstance(department_id, int)
        or department_id <= 0
    ):
        raise ValueError("department_id must be a positive integer")

    conn = None
    try:
        conn = create_db_connection()
        cursor = conn.cursor()
        department_predicate = "dep.dept_id = ?" if department_id is not None else "dep.name = ?"
        department_parameter = department_id if department_id is not None else department_name
        cursor.execute(f"""
            SELECT DISTINCT
                doc.doctor_id,
                doc.name
            FROM Doctor doc
            JOIN Schedule s ON s.doctor_id = doc.doctor_id
            JOIN Department dep ON dep.dept_id = s.dept_id
            WHERE {department_predicate}
                AND doc.is_active = 1
                AND doc.is_placeholder = 0
            ORDER BY doc.name, doc.doctor_id
        """, department_parameter)
        doctors = []
        seen_ids = set()
        for row in cursor.fetchall():
            doctor_id = str(row[0])
            name = str(row[1] or "").strip()
            if not name or doctor_id in seen_ids:
                continue
            seen_ids.add(doctor_id)
            doctors.append({"doctor_id": doctor_id, "name": name})
        return doctors
    except Exception as exc:
        raise DatabaseUnavailableError("正式醫師資料目前無法載入。") from exc
    finally:
        if conn is not None:
            conn.close()


def fetch_return_visit_slots(
    department_name: str,
    doctor_name: str,
    preferred_dates: List[str],
    preferred_sessions: List[str],
    *,
    department_id: int | None = None,
    doctor_id: int | None = None,
) -> List[dict]:
    """Fetch exact return-visit schedules without interpreting canonical visit_type.

    The relationship CTE intentionally has no date restriction. Schedule availability is
    then queried by the selected doctor/department identity, date, and optional session.
    """
    dates = [date.fromisoformat(value) for value in preferred_dates]
    search_start = datetime.now(TAIPEI_ZONE).date()

    normalized_sessions = {
        _normalize_schedule_session(value)
        for value in preferred_sessions
        if str(value or "").strip()
    }
    normalized_sessions.discard("")

    identity_clauses = []
    identity_params: list[Any] = [department_name, doctor_name]
    if department_id is not None:
        identity_clauses.append("AND dep.dept_id = ?")
        identity_params.append(department_id)
    if doctor_id is not None:
        identity_clauses.append("AND doc.doctor_id = ?")
        identity_params.append(doctor_id)

    if dates:
        date_placeholders = ", ".join("?" for _ in dates)
        date_clause = f"s.date IN ({date_placeholders})"
        date_params: list[Any] = list(dates)
    else:
        date_clause = "s.date >= ? AND s.date <= DATEADD(day, 20, ?)"
        date_params = [search_start, search_start]
    session_clause = ""
    session_params: list[Any] = []
    if normalized_sessions:
        session_expressions = []
        for session in sorted(normalized_sessions):
            aliases = _schedule_session_aliases(session)
            alias_placeholders = ", ".join("?" for _ in aliases)
            session_expressions.append(f"LTRIM(RTRIM(s.session)) IN ({alias_placeholders})")
            session_params.extend(aliases)
        session_clause = "AND (" + " OR ".join(session_expressions) + ")"

    query = f"""
        WITH SelectedDoctorDepartment AS (
            SELECT DISTINCT
                assoc.doctor_id,
                assoc.dept_id
            FROM Schedule assoc
            JOIN Doctor doc ON assoc.doctor_id = doc.doctor_id
            JOIN Department dep ON assoc.dept_id = dep.dept_id
            WHERE dep.name = ?
                AND doc.name = ?
                AND doc.is_active = 1
                AND doc.is_placeholder = 0
                {' '.join(identity_clauses)}
        )
        SELECT
            dc.name AS parent_dept,
            dep.name AS child_dept,
            doc.name AS doctor_name,
            s.session,
            doc.doctor_id,
            s.schedule_id,
            s.room,
            doc.specialty_tags,
            s.status,
            s.date,
            s.visit_type,
            doc.is_placeholder,
            dep.dept_id
        FROM Schedule s
        JOIN SelectedDoctorDepartment selected
            ON s.doctor_id = selected.doctor_id
            AND s.dept_id = selected.dept_id
        JOIN Doctor doc ON s.doctor_id = doc.doctor_id
        JOIN Department dep ON s.dept_id = dep.dept_id
        JOIN DepartmentCategory dc ON dep.category_id = dc.category_id
        WHERE {date_clause}
            AND doc.is_active = 1
            AND doc.is_placeholder = 0
            AND (
                s.status IS NULL
                OR LOWER(LTRIM(RTRIM(s.status))) IN ('open', 'available', N'可掛號')
            )
            {session_clause}
        ORDER BY s.date ASC, s.session ASC, s.schedule_id ASC
    """

    conn = None
    try:
        conn = create_db_connection()
        cursor = conn.cursor()
        cursor.execute(query, *(identity_params + date_params + session_params))
        rows = cursor.fetchall()
    except Exception as exc:
        raise DatabaseUnavailableError("正式回診班表目前無法查詢。") from exc
    finally:
        if conn is not None:
            conn.close()

    slots = []
    for row in rows:
        raw_status = row[8]
        if bool(row[11]) or not _schedule_status_is_available(raw_status):
            continue
        slot = {
            "parent_dept": str(row[0] or "").strip(),
            "child_dept": str(row[1] or "").strip(),
            "doctor": str(row[2] or "").strip(),
            "session": str(row[3] or "").strip(),
            "doctor_id": str(row[4]),
            "schedule_id": str(row[5]),
            "room": row[6] or "",
            "slot": row[6] or "",
            "specialty_tags": row[7] or "",
            "status": "open",
            "status_raw": raw_status or "",
            "is_leave": False,
            "substitute_doctor": "",
            "date": row[9],
            "visit_type": str(row[10] or "").strip(),
            "is_placeholder": False,
            "dept_id": str(row[12]),
            "source": "db",
            "source_detail": "SQL Server Schedule",
        }
        if slot["child_dept"] != department_name or slot["doctor"] != doctor_name:
            continue
        if department_id is not None and slot["dept_id"] != str(department_id):
            continue
        if doctor_id is not None and slot["doctor_id"] != str(doctor_id):
            continue
        if dates and slot["date"] not in dates:
            continue
        slot_date = slot["date"].date() if isinstance(slot["date"], datetime) else slot["date"]
        if not dates and not (search_start <= slot_date <= search_start + timedelta(days=20)):
            continue
        if normalized_sessions and _normalize_schedule_session(slot["session"]) not in normalized_sessions:
            continue
        slots.append(slot)
    return slots


def _normalize_schedule_session(value: Any) -> str:
    text = str(value or "").strip()
    aliases = {
        "早上": "上午",
        "上午": "上午",
        "morning": "上午",
        "下午": "下午",
        "午診": "下午",
        "afternoon": "下午",
        "晚上": "晚上",
        "晚診": "晚上",
        "夜診": "晚上",
        "夜間": "晚上",
        "evening": "晚上",
    }
    return aliases.get(text, text)


def _schedule_session_aliases(normalized: str) -> list[str]:
    return {
        "上午": ["上午", "早上", "morning"],
        "下午": ["下午", "午診", "afternoon"],
        "晚上": ["晚上", "晚診", "夜診", "夜間", "evening"],
    }.get(normalized, [normalized])


def _normalized_schedule_session_sql(alias: str) -> str:
    value = f"LOWER(LTRIM(RTRIM(COALESCE({alias}.session, N''))))"
    return f"""
        CASE
            WHEN {value} IN (N'早上', N'上午', N'morning') THEN N'上午'
            WHEN {value} IN (N'下午', N'午診', N'afternoon') THEN N'下午'
            WHEN {value} IN (N'晚上', N'晚診', N'夜診', N'夜間', N'evening') THEN N'晚上'
            ELSE {value}
        END
    """


def _quick_search_followup_evidence_sql(alias: str) -> str:
    selected_session = _normalized_schedule_session_sql(alias)
    companion_session = _normalized_schedule_session_sql("followup")
    return f"""
        CASE WHEN
            {alias}.visit_type IS NULL
            OR LTRIM(RTRIM({alias}.visit_type)) = N''
            OR EXISTS (
                SELECT 1
                FROM Schedule followup
                WHERE followup.doctor_id = {alias}.doctor_id
                    AND followup.dept_id = {alias}.dept_id
                    AND followup.date = {alias}.date
                    AND {companion_session} = {selected_session}
                    AND LTRIM(RTRIM(COALESCE(followup.room, N''))) =
                        LTRIM(RTRIM(COALESCE({alias}.room, N'')))
                    AND LTRIM(RTRIM(followup.visit_type)) = ?
            )
        THEN 1 ELSE 0 END
    """


def fetch_quick_search_slots(
    department_id: int,
    target_date: date,
    period: str,
) -> List[dict]:
    """Fetch available schedules with evidence that the physical slot supports follow-up.

    Exceptions are allowed to propagate so the API can report DB unavailability without
    substituting mock data.
    """
    normalized_period = _normalize_schedule_session(period)
    aliases = _schedule_session_aliases(normalized_period)
    session_placeholders = ", ".join("?" for _ in aliases)
    followup_evidence = _quick_search_followup_evidence_sql("s")
    query = f"""
        SELECT
            dc.name AS parent_dept,
            dep.name AS child_dept,
            doc.name AS doctor_name,
            s.session,
            doc.doctor_id,
            s.schedule_id,
            s.room,
            doc.specialty_tags,
            s.status,
            s.date,
            s.visit_type,
            doc.is_placeholder,
            dep.dept_id,
            {followup_evidence} AS supports_followup
        FROM Schedule s
        JOIN Doctor doc ON s.doctor_id = doc.doctor_id
        JOIN Department dep ON s.dept_id = dep.dept_id
        JOIN DepartmentCategory dc ON dep.category_id = dc.category_id
        WHERE dep.dept_id = ?
            AND s.date = ?
            AND doc.is_active = 1
            AND doc.is_placeholder = 0
            AND (
                s.status IS NULL
                OR LOWER(LTRIM(RTRIM(s.status))) IN ('open', 'available', N'可掛號')
            )
            AND LTRIM(RTRIM(s.session)) IN ({session_placeholders})
        ORDER BY doc.name ASC, s.schedule_id ASC
    """

    conn = None
    try:
        conn = create_db_connection()
        cursor = conn.cursor()
        cursor.execute(query, QUICK_SEARCH_VISIT_TYPE, department_id, target_date, *aliases)
        rows = cursor.fetchall()
    except Exception as exc:
        raise DatabaseUnavailableError("正式班表目前無法查詢。") from exc
    finally:
        if conn is not None:
            conn.close()

    slots = []
    seen = set()
    for row in rows:
        raw_status = row[8]
        if bool(row[11]) or not _schedule_status_is_available(raw_status):
            continue
        doctor_name = str(row[2] or "").strip()
        schedule_id = str(row[5]) if row[5] is not None else ""
        if not doctor_name:
            continue
        key = schedule_id or (str(row[4]), str(row[9]), str(row[3] or ""), str(row[6] or ""))
        if key in seen:
            continue
        seen.add(key)
        slots.append(
            {
                "parent_dept": str(row[0] or "").strip(),
                "child_dept": str(row[1] or "").strip(),
                "doctor": doctor_name,
                "session": str(row[3] or "").strip(),
                "doctor_id": str(row[4]) if row[4] is not None else "",
                "schedule_id": schedule_id,
                "room": row[6] or "",
                "slot": row[6] or "",
                "specialty_tags": row[7] or "",
                "status": "open",
                "status_raw": raw_status or "",
                "is_leave": False,
                "substitute_doctor": "",
                "date": row[9],
                "visit_type": str(row[10] or "").strip(),
                "is_placeholder": False,
                "dept_id": str(row[12]),
                "supports_followup": bool(row[13]),
                "source": "db",
                "source_detail": "SQL Server Schedule",
            }
        )
    return slots


def fetch_quick_search_slot_by_id(schedule_id: str) -> dict | None:
    """Reload one available schedule and its physical-slot follow-up evidence."""
    followup_evidence = _quick_search_followup_evidence_sql("s")
    query = f"""
        SELECT
            dc.name AS parent_dept,
            dep.name AS child_dept,
            doc.name AS doctor_name,
            s.session,
            doc.doctor_id,
            s.schedule_id,
            s.room,
            doc.specialty_tags,
            s.status,
            s.date,
            s.visit_type,
            doc.is_placeholder,
            dep.dept_id,
            {followup_evidence} AS supports_followup
        FROM Schedule s
        JOIN Doctor doc ON s.doctor_id = doc.doctor_id
        JOIN Department dep ON s.dept_id = dep.dept_id
        JOIN DepartmentCategory dc ON dep.category_id = dc.category_id
        WHERE s.schedule_id = ?
            AND doc.is_active = 1
            AND doc.is_placeholder = 0
            AND (
                s.status IS NULL
                OR LOWER(LTRIM(RTRIM(s.status))) IN ('open', 'available', N'可掛號')
            )
    """

    conn = None
    try:
        conn = create_db_connection()
        cursor = conn.cursor()
        cursor.execute(query, QUICK_SEARCH_VISIT_TYPE, schedule_id)
        rows = cursor.fetchall()
    except Exception as exc:
        raise DatabaseUnavailableError("正式班表目前無法重新確認。") from exc
    finally:
        if conn is not None:
            conn.close()

    if not rows:
        return None
    row = rows[0]
    raw_status = row[8]
    doctor_name = str(row[2] or "").strip()
    if bool(row[11]) or not doctor_name or not _schedule_status_is_available(raw_status):
        return None
    return {
        "parent_dept": str(row[0] or "").strip(),
        "child_dept": str(row[1] or "").strip(),
        "doctor": doctor_name,
        "session": str(row[3] or "").strip(),
        "doctor_id": str(row[4]) if row[4] is not None else "",
        "schedule_id": str(row[5]) if row[5] is not None else "",
        "room": row[6] or "",
        "slot": row[6] or "",
        "specialty_tags": row[7] or "",
        "status": "open",
        "status_raw": raw_status or "",
        "is_leave": False,
        "substitute_doctor": "",
        "date": row[9],
        "visit_type": str(row[10] or "").strip(),
        "is_placeholder": False,
        "dept_id": str(row[12]),
        "supports_followup": bool(row[13]),
        "source": "db",
        "source_detail": "SQL Server Schedule",
    }


def fetch_available_slots(
    department_text: str | None,
    search_days: int = 14,
    max_slots: int | None = 10,
    schedule_visit_type: str | None = None,
    *,
    department_id: int | None = None,
) -> List[dict]:
    _set_last_db_fallback_reason("")
    try:
        slots = _fetch_available_slots_from_db(
            department_text,
            search_days,
            max_slots,
            schedule_visit_type=schedule_visit_type,
            department_id=department_id,
        )
        if slots:
            return slots
        reason = _last_db_fallback_reason or "no rows returned"
        if schedule_visit_type:
            logger.info(
                "DB schedule has no compatible slots department=%s visit_type=%s reason=%s",
                department_id if department_id is not None else department_text,
                schedule_visit_type,
                reason,
            )
            return []
        logger.info("DB schedule returned no usable slots department_id=%s reason=%s", department_id, reason)
    except DatabaseUnavailableError:
        raise
    except Exception as exc:
        logger.error(
            "DB schedule query failed department_id=%s error_type=%s",
            department_id,
            type(exc).__name__,
        )
        raise DatabaseUnavailableError("正式班表目前無法查詢。") from exc
    return []


def _fetch_available_slots_from_db(
    department_text: str | None,
    search_days: int,
    max_slots: int | None,
    schedule_visit_type: str | None = None,
    *,
    department_id: int | None = None,
) -> List[dict]:
    _set_last_db_fallback_reason("")
    conn = None

    visit_type_clause = " AND s.visit_type = ?" if schedule_visit_type else ""
    department_clause = "dep.dept_id = ?" if department_id is not None else "dep.name LIKE ?"
    query = f"""
        SELECT 
            dc.name AS parent_dept,
            dep.name AS child_dept,
            doc.name AS doctor_name,
            s.session,
            doc.doctor_id,
            s.schedule_id,
            s.room,
            doc.specialty_tags,
            s.status,
            s.date,
            s.visit_type,
            doc.is_placeholder,
            dep.dept_id
        FROM Schedule s
        JOIN Doctor doc ON s.doctor_id = doc.doctor_id
        JOIN Department dep ON s.dept_id = dep.dept_id
        JOIN DepartmentCategory dc ON dep.category_id = dc.category_id
        WHERE {department_clause}
            AND s.date >= ?
            AND s.date <= DATEADD(day, ?, ?)
            AND doc.is_active = 1
            {visit_type_clause}
        ORDER BY s.date ASC, s.session ASC
    """

    start_date = datetime.now(TAIPEI_ZONE).date()
    department_param: Any = department_id if department_id is not None else f"%{department_text or ''}%"
    params: list[Any] = [department_param, start_date, search_days, start_date]
    if schedule_visit_type:
        params.append(schedule_visit_type)
    try:
        conn = create_db_connection()
        cursor = conn.cursor()
        cursor.execute(query, *params)
        rows = cursor.fetchall()
    except Exception as exc:
        raise DatabaseUnavailableError("正式班表目前無法查詢。") from exc
    finally:
        if conn is not None:
            conn.close()
    logger.info(
        "[DB_QUERY] department_id=%s date_from=%s date_to=%s visit_type=%s "
        "raw_schedule_count=%s",
        department_id,
        start_date,
        start_date + timedelta(days=search_days),
        schedule_visit_type or "all",
        len(rows),
    )

    if not rows:
        _set_last_db_fallback_reason("no rows returned")
        return []

    slots = []
    seen = {}
    unavailable_count = 0
    placeholder_count = 0
    for row in rows:
        raw_status = row[8]
        is_placeholder = bool(row[11])
        if is_placeholder:
            placeholder_count += 1
            continue
        if not _schedule_status_is_available(raw_status):
            unavailable_count += 1
            continue

        key = (str(row[4]), row[9], str(row[3] or ""), str(row[6] or ""))
        visit_type = str(row[10] or "").strip()
        if schedule_visit_type and visit_type != schedule_visit_type:
            continue
        if key in seen:
            _merge_visit_type(seen[key], visit_type)
            continue

        slot = {
            "parent_dept": row[0],
            "child_dept": row[1],
            "dept_id": int(row[12]) if row[12] is not None else None,
            "doctor": row[2],
            "session": row[3],
            "doctor_id": str(row[4]) if row[4] is not None else "",
            "schedule_id": str(row[5]) if row[5] is not None else "",
            "room": row[6] or "",
            "slot": row[6] or "",
            "specialty_tags": row[7] or "",
            "status": "open",
            "status_raw": raw_status or "",
            "is_leave": False,
            "substitute_doctor": "",
            "date": row[9],
            "visit_type": visit_type,
            "is_placeholder": False,
            "source": "db",
            "source_detail": "SQL Server Schedule",
        }
        seen[key] = slot
        slots.append(slot)
    if not slots:
        if unavailable_count and unavailable_count + placeholder_count == len(rows):
            _set_last_db_fallback_reason("all rows unavailable by status")
        elif placeholder_count == len(rows):
            _set_last_db_fallback_reason("all rows are placeholder doctors")
        else:
            _set_last_db_fallback_reason("no usable rows returned")
    return slots if max_slots is None else slots[:max_slots]


def map_schedule_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "parent_dept": record.get("parent_dept") or record.get("parentDept") or "",
        "child_dept": record.get("child_dept") or record.get("childDept") or "",
        "dept_id": record.get("dept_id") or record.get("deptId"),
        "doctor_id": str(record.get("doctor_id") or record.get("doctorId") or record.get("doctor") or ""),
        "doctor": record.get("doctor") or record.get("doctor_name") or record.get("doctorName") or "",
        "schedule_id": str(record.get("schedule_id") or record.get("scheduleId") or ""),
        "date": record.get("date") or "",
        "session": record.get("session") or "",
        "slot": record.get("slot") or record.get("room") or "",
        "room": record.get("room") or record.get("slot") or "",
        "specialty_tags": record.get("specialty_tags") or record.get("specialtyTags") or record.get("specialty") or "",
        "status": record.get("status") or "open",
        "is_leave": bool(record.get("is_leave", False)),
        "substitute_doctor": record.get("substitute_doctor") or record.get("substituteDoctor") or "",
        "source": record.get("source") or "db",
        "source_detail": record.get("source_detail") or record.get("sourceDetail") or "",
        "fallback_reason": record.get("fallback_reason") or record.get("fallbackReason") or "",
        "visit_type": record.get("visit_type") or record.get("visitType") or "",
        "status_raw": record.get("status_raw") or record.get("statusRaw") or "",
    }
