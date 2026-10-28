import json
from predict import predict_requirement
from conflict_detector import detect_conflicts

fr_statements = [
    "FR-01: The system shall allow patients to create an account using their name, mobile number, email address, and password.",
    "FR-02: The system shall allow registered patients to log in using their email address and password.",
    "FR-03: The system shall allow patients to search for doctors by doctor name, department, specialization, and availability.",
    "FR-04: The system shall allow a patient to book an appointment with an available doctor for a selected date and time slot.",
    "FR-05: The system shall allow patients to cancel an appointment before the consultation begins.",
    "FR-06: The system shall send an appointment confirmation notification to the patient after a booking is successfully created.",
    "FR-07: The system shall send appointment reminders to patients at an appropriate time before their scheduled consultation.",
    "FR-08: The system shall allow doctors to view the appointments assigned to them and update the consultation status.",
    "FR-09: The system shall allow doctors to record consultation notes and prescriptions for a patient.",
    "FR-10: The system shall allow authorized hospital staff to view a patient's previous consultation records.",
    "FR-11: The system shall allow administrators to add, edit, deactivate, and reactivate doctor profiles.",
    "FR-12: The system shall allow administrators to generate reports containing appointment activity and doctor utilization.",
    "FR-13: The system shall allow patients to receive notifications through email and mobile notification channels.",
    "FR-14: The system shall allow only authorized medical staff to access patient medical records.",
    "FR-15: The system shall allow all registered users to view a patient's medical record when the patient's name is known.",
    "FR-16: The system shall prevent two patients from booking the same doctor and time slot.",
    "FR-17: The system shall permit patients to book any doctor even when the selected doctor has no available appointment slot.",
    "FR-18: The system shall provide a fast response to patient searches.",
    "FR-19: The system shall maintain a secure environment for patient information and authentication credentials.",
    "FR-20: The system shall be available to users throughout the day with minimal interruption.",
    "BR-04: The system shall automatically remove old notifications after a reasonable period."
]

print("=== AMBIGUITY DETECTION TEST ON CAMPUSCARE SRS REQUIREMENTS ===")
amb_count = 0
for text in fr_statements:
    res = predict_requirement(text)
    pred = res['prediction']
    if pred == 'Ambiguous':
        amb_count += 1
    print(f"{text[:6]} -> Prediction: {pred:<10} | Confidence: {res['confidence']:<4} | Reason: {res['reason']}")

print(f"\nTotal Ambiguous Detected: {amb_count} / {len(fr_statements)}")

print("\n=== CONFLICT DETECTION TEST ON CAMPUSCARE SRS REQUIREMENTS ===")
conf_res = detect_conflicts(fr_statements)
conf_list = conf_res.get("conflicts", [])
print(f"Total Conflicts Found: {len(conf_list)}")
print("Conflicts Detail:")
for i, c in enumerate(conf_list, 1):
    r1_id = c.get('requirement_1_id') or c.get('req_a_id')
    r1_txt = c.get('requirement_1_text') or c.get('req_a_text')
    r2_id = c.get('requirement_2_id') or c.get('req_b_id')
    r2_txt = c.get('requirement_2_text') or c.get('req_b_text')
    print(f"  {i}. [{r1_id}] vs [{r2_id}] | Severity: {c.get('severity')} | Type: {c.get('conflict_type')}")
    print(f"     Req 1: {r1_txt}")
    print(f"     Req 2: {r2_txt}\n")
