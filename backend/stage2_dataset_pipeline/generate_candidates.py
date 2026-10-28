"""
ARIP Stage 2 Pipeline: Parts 9, 10, 11, 12, 13, 14 — Targeted Synthetic Candidate Generator (generate_candidates.py)

Generates targeted, domain-diverse synthetic requirement (YES) and non-requirement (NO)
candidates to fill category representation gaps.

Key Principles:
1. 20+ Software Domains: Healthcare, Banking, Education, E-Commerce, Transportation, Hospital,
   Library, Hotel, Insurance, Manufacturing, Government, Social Media, Logistics, Cybersecurity,
   Cloud, Mobile, IoT, Finance, HR, Inventory.
2. Phrasing Styles: Explicit, Modal, User-Oriented, Conditional, Business Rule, Performance,
   Security, Passive, Imperative.
3. Hard Negatives: Implementation, architecture, completion, and discussion statements that
   resemble requirements but lack requirement intent.
Outputs: data/generated/synthetic_candidates.csv
"""

import os
import json
import random
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATED_DIR = os.path.join(BASE_DIR, "data", "generated")
SYNTHETIC_CSV_PATH = os.path.join(GENERATED_DIR, "synthetic_candidates.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "data", "reports")
GEN_REPORT_PATH = os.path.join(REPORTS_DIR, "generation_report.json")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

DOMAINS = [
    "Healthcare", "Banking", "Education", "E-Commerce", "Transportation",
    "Hospital Management", "Library Management", "Hotel Management", "Insurance",
    "Manufacturing", "Government", "Social Media", "Logistics", "Cybersecurity",
    "Cloud Applications", "Mobile Applications", "IoT", "Finance", "HR Systems", "Inventory Systems"
]

PHRASING_STYLES = [
    "Explicit ('shall')", "Modal ('must/should')", "User-Oriented ('Users can')",
    "Conditional ('If X occurs...')", "Business Rule ('Only verified...')",
    "Performance ('within X ms')", "Security ('encrypt...')", "Passive ('Data shall be...')",
    "Imperative ('Allow admins to...')"
]

# Rich patterns for diverse generation
YES_PATTERNS = {
    "Security": [
        "The {domain} application shall encrypt all sensitive customer data at rest using AES-{keylen} encryption.",
        "System administrators must enforce multi-factor authentication for all remote access logins in the {domain} portal.",
        "The {domain} platform shall automatically log out inactive user sessions after {num} minutes.",
        "The system must validate digital signatures on all uploaded {domain} documents before processing.",
        "The {domain} module shall restrict access to administrative tools to authorized security personnel only.",
        "Passwords in the {domain} database must be hashed using bcrypt with a salt factor of at least {saltlen}.",
        "The system shall mask all credit card and national ID numbers in {domain} view screens.",
        "API calls in the {domain} service must require a valid OAuth 2.0 bearer token for authorization.",
        "The {domain} system shall lock out user accounts after {num} failed authentication attempts within {num} minutes.",
        "The application must maintain immutable audit logs of all privilege escalation events in the {domain} portal."
    ],
    "Performance": [
        "The {domain} system shall process peak transactions within {num} seconds of user submission.",
        "The {domain} search engine must return query results within {num0} milliseconds for up to {num000} concurrent users.",
        "The system shall maintain an application page load time of under {num} seconds on mobile networks.",
        "The {domain} database must execute read queries within {num0} ms under standard operational load.",
        "The platform shall support up to {num000} simultaneous live connections without performance degradation.",
        "The {domain} reporting engine must generate batch export files within {num} minutes for datasets under 100,000 rows.",
        "The system shall handle at least {num000} concurrent API requests per second in the {domain} service."
    ],
    "Authentication": [
        "Users shall be able to log in to the {domain} portal using valid email and password credentials.",
        "The {domain} application must send a one-time verification code via SMS during new account registration.",
        "The system shall lock out any {domain} user account after {num} consecutive failed authentication attempts.",
        "Users must be able to reset a forgotten password using a secure email link valid for {num0} minutes.",
        "The {domain} mobile app shall support biometric authentication via fingerprint and facial recognition.",
        "The system shall issue JWT session tokens with a maximum lifespan of {num} hours upon successful {domain} login."
    ],
    "Access Control": [
        "The {domain} application shall enforce role-based access control for all system endpoints.",
        "Supervisors must be able to grant or revoke operational privileges for {domain} staff members.",
        "The system shall restrict access to confidential {domain} records based on user department assignment.",
        "Only authorized finance officers shall be able to approve payments exceeding ${num000} in the {domain} system.",
        "The {domain} system shall restrict deletion of historical audit logs to system superadministrators."
    ],
    "Availability": [
        "The {domain} platform shall maintain {percent}% operational uptime during peak business hours.",
        "The system must support failover to a redundant secondary server within {num0} seconds of primary server outage.",
        "The {domain} service shall operate continuously 24/7 with scheduled maintenance limited to off-peak hours."
    ],
    "Reliability": [
        "The {domain} database shall execute automated incremental backups every {num} hours.",
        "The system must restore complete operational state from backup within {num} hours of a disaster event.",
        "The {domain} transaction pipeline shall execute atomic database commits to prevent data corruption."
    ],
    "Usability": [
        "The {domain} user interface shall comply with WCAG 2.1 AA accessibility guidelines.",
        "The application must support full keyboard navigation for all interactive forms in the {domain} portal.",
        "The {domain} system shall display clear error messages explaining invalid input formatting to users."
    ],
    "API/Interface": [
        "The {domain} API shall return standardized JSON formatted error messages with HTTP status codes.",
        "The system must integrate with third-party {domain} payment gateways using secure REST web services.",
        "The {domain} platform shall expose open OpenAPI v3 compliant REST endpoints for external integrations."
    ],
    "Reporting": [
        "The {domain} system shall generate monthly performance summary reports exportable in PDF and Excel formats.",
        "Managers must be able to view real-time analytical dashboards showing daily {domain} metrics.",
        "The system shall compile quarterly auditing reports displaying department-level transaction totals."
    ],
    "Notification": [
        "The {domain} application shall send automated email notifications when an order status changes to shipped.",
        "The system must alert administrators via SMS when server CPU utilization exceeds {percent}%.",
        "Users shall receive push notifications on mobile devices 2 hours before scheduled {domain} events."
    ],
    "Business Rules": [
        "The {domain} system shall calculate a {num}% discount for customer orders exceeding ${num00}.",
        "Late payment penalties shall be calculated at {num}% per month on overdue {domain} invoice balances.",
        "The system must automatically apply tax rates based on the customer's billing country in the {domain} checkout."
    ],
    "Validation": [
        "The {domain} system shall validate email address formatting using standard RFC regex checks.",
        "The application must verify that uploaded file attachments do not exceed {num} MB in size.",
        "The {domain} form shall prevent submission if required fields are left blank by the user."
    ],
    "Functional": [
        "Customers shall be able to add selected items to a digital shopping cart in the {domain} store.",
        "The {domain} system shall allow users to upload profile pictures in PNG or JPG format.",
        "Staff shall be able to create, update, and archive customer accounts in the {domain} management system.",
        "The application shall display real-time inventory availability for all listed {domain} items.",
        "Users can filter search results by date range, category, and price in the {domain} portal."
    ]
}

NO_PATTERNS = {
    "Section Heading": [
        "1.0 INTRODUCTION AND PURPOSE OF THE SYSTEM",
        "2.3 SYSTEM ARCHITECTURE AND COMPONENT OVERVIEW",
        "SECTION 3: FUNCTIONAL REQUIREMENTS SPECIFICATION",
        "4.1 HARDWARE INTERFACE SPECIFICATIONS",
        "APPENDIX A: GLOSSARY OF TERMS AND ACRONYMS",
        "REVISION HISTORY AND DOCUMENT CONTROL LOG",
        "5.0 NON-FUNCTIONAL SYSTEM PERFORMANCE REQUIREMENTS"
    ],
    "Introduction": [
        "This document describes the software requirements specification for the {domain} project.",
        "The purpose of this document is to outline the operational goals of the new {domain} software.",
        "This specification is intended for software developers, testers, and project stakeholders."
    ],
    "Background": [
        "The legacy {domain} application was developed in 2012 and suffers from poor maintainability.",
        "Currently, manual paper forms are used to manage {domain} records across branch offices.",
        "Due to increasing transaction volume, the existing {domain} database has reached storage capacity limits."
    ],
    "System Overview": [
        "The {domain} platform consists of three main modules: a web frontend, a REST API, and a database.",
        "The application architecture uses a microservices model deployed on cloud infrastructure.",
        "Overall, the {domain} solution connects end-users, administrative staff, and third-party vendors."
    ],
    "Figure/Table Caption": [
        "Figure 3.1: High-level architectural diagram of the {domain} web application.",
        "Table 4.2: Data dictionary mapping for {domain} user account fields.",
        "Figure 2.5: Sequence diagram illustrating user authentication in the {domain} portal."
    ],
    "Glossary/Reference": [
        "IEEE Std 830-1998: Recommended Practice for Software Requirements Specifications.",
        "AES: Advanced Encryption Standard, a symmetric-key encryption algorithm.",
        "REST: Representational State Transfer architectural style for distributed hypermedia systems."
    ],
    "Implementation/Design Description": [
        "The backend service was implemented in Python using the FastAPI framework.",
        "The authentication module uses JSON Web Tokens (JWT) for stateless session management.",
        "We selected PostgreSQL as the primary relational database management system for the {domain} project.",
        "The user interface was built using React and TypeScript with Tailwind CSS styling."
    ],
    "Document Metadata": [
        "Submitted By: Project Team Lead (Computer Engineering Department).",
        "Guided By: Dr. A. K. Sharma (Associate Professor, Department of Computer Science).",
        "Document Version 2.4 - Final Approved Build - Academic Year 2025-2026."
    ],
    "Hard Negatives (Requirement-Like NO)": [
        "Authentication is an important feature of the modern {domain} web platform.",
        "The {domain} application provides user management capabilities for administrative staff.",
        "The system has been updated to support mobile browser accessibility.",
        "Payment integration allows users to complete checkout using credit cards or digital wallets.",
        "The reporting module was designed to export financial data into Excel spreadsheets.",
        "Data security is critical to protecting customer privacy in the {domain} database.",
        "The search feature helps users locate historical records in the system.",
        "Notifications inform users when their account profile has been modified.",
        "File upload functionality supports images up to 5 MB in file size.",
        "Password verification ensures that account credentials meet security standards.",
        "Session management allows users to maintain active connections across page refreshes.",
        "Audit logging provides a complete record of system activity for administrative review."
    ]
}


def generate_synthetic_dataset() -> pd.DataFrame:
    print("=" * 70)
    print("      ARIP STAGE 2: PARTS 9-14 — TARGETED SYNTHETIC GENERATION      ")
    print("=" * 70)

    generated_rows = []
    sample_id_counter = 1

    # Unique generation targeting ~3,200 YES and ~3,200 NO candidates
    yes_categories = list(YES_PATTERNS.keys())
    no_categories = list(NO_PATTERNS.keys())

    seen_generated_texts = set()

    # Generate YES candidates
    for _ in range(3500):
        cat = random.choice(yes_categories)
        pattern = random.choice(YES_PATTERNS[cat])
        domain = random.choice(DOMAINS)
        style = random.choice(PHRASING_STYLES)

        text = pattern.format(
            domain=domain,
            num=random.choice([2, 3, 5, 10, 15, 30, 60]),
            num0=random.choice([100, 200, 300, 500]),
            num00=random.choice([100, 250, 500, 1000]),
            num000=random.choice([1000, 2500, 5000, 10000]),
            percent=random.choice([95, 98, 99, 99.9]),
            keylen=random.choice([128, 256]),
            saltlen=random.choice([10, 12, 14])
        )

        norm_t = text.lower()
        if norm_t in seen_generated_texts:
            continue

        seen_generated_texts.add(norm_t)

        generated_rows.append({
            "candidate_id": f"SYN_{sample_id_counter:05d}",
            "text": text,
            "proposed_label": "YES",
            "category": cat,
            "domain": domain,
            "style": style,
            "source": "synthetic_generator"
        })
        sample_id_counter += 1

    # Generate NO candidates
    for _ in range(3500):
        cat = random.choice(no_categories)
        pattern = random.choice(NO_PATTERNS[cat])
        domain = random.choice(DOMAINS)

        text = pattern.format(domain=domain)
        norm_t = text.lower()
        if norm_t in seen_generated_texts:
            continue

        seen_generated_texts.add(norm_t)

        generated_rows.append({
            "candidate_id": f"SYN_{sample_id_counter:05d}",
            "text": text,
            "proposed_label": "NO",
            "category": cat,
            "domain": domain,
            "style": "Declarative/Informative",
            "source": "synthetic_generator"
        })
        sample_id_counter += 1

    df_gen = pd.DataFrame(generated_rows)
    
    os.makedirs(GENERATED_DIR, exist_ok=True)
    df_gen.to_csv(SYNTHETIC_CSV_PATH, index=False)

    gen_report = {
        "total_generated": len(df_gen),
        "generated_yes_count": sum(df_gen["proposed_label"] == "YES"),
        "generated_no_count": sum(df_gen["proposed_label"] == "NO"),
        "domain_distribution": df_gen["domain"].value_counts().to_dict(),
        "category_distribution": df_gen["category"].value_counts().to_dict(),
        "style_distribution": df_gen["style"].value_counts().to_dict()
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(GEN_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(gen_report, f, indent=4)

    print(f"[+] Synthetic generation complete. Saved candidates to: {SYNTHETIC_CSV_PATH}")
    print(f"    - Generated Total: {len(df_gen)} (YES: {gen_report['generated_yes_count']}, NO: {gen_report['generated_no_count']})")
    print(f"    - Software Domains Covered: {len(DOMAINS)}")
    print("=" * 70)

    return df_gen


if __name__ == "__main__":
    generate_synthetic_dataset()
