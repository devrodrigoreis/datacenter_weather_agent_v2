#!/usr/bin/env python3
"""
Reprocess existing violation logs to update insights.
This will clean up the threat types and regenerate insights.
"""

import json
from pathlib import Path
from datetime import datetime

def clean_threat_type(threat_type: str) -> str:
    """Clean up threat type to just the label."""
    # Remove everything after first newline
    threat_type = threat_type.split('\n')[0]
    # Remove common prefixes
    threat_type = threat_type.lstrip('- ').lstrip('* ').strip()
    # Remove any trailing punctuation or notes
    threat_type = threat_type.split('.')[0].split(',')[0].split(':')[0].strip()
    # Ensure lowercase
    threat_type = threat_type.lower()
    
    # Validate and map to valid labels
    valid_labels = {
        'prompt_extraction': ['prompt_extraction', 'prompt', 'extraction'],
        'credential_extraction': ['credential_extraction', 'credential', 'password'],
        'role_manipulation': ['role_manipulation', 'role', 'manipulation'],
        'security_bypass': ['security_bypass', 'security', 'bypass'],
        'config_inspection': ['config_inspection', 'config', 'inspection', 'internal']
    }
    
    # Try exact match first
    if threat_type in valid_labels:
        return threat_type
    
    # Try fuzzy match
    for valid_label, keywords in valid_labels.items():
        for keyword in keywords:
            if keyword in threat_type or threat_type in keyword:
                return valid_label
    
    # Default
    return 'config_inspection'

def main():
    log_dir = Path("security_logs")
    today = datetime.now().strftime('%Y%m%d')
    log_file = log_dir / f"violations_{today}.jsonl"
    
    if not log_file.exists():
        print(f"No violations file found: {log_file}")
        return
    
    # Read all violations
    violations = []
    with open(log_file, "r") as f:
        for line in f:
            try:
                violations.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    print(f"Found {len(violations)} violations")
    
    # Clean and update violations file
    cleaned_violations = []
    for v in violations:
        original_threat = v.get("threat_type", "unknown")
        cleaned_threat = clean_threat_type(original_threat)
        v["threat_type"] = cleaned_threat
        cleaned_violations.append(v)
        print(f"  Cleaned: {original_threat[:50]}... -> {cleaned_threat}")
    
    # Rewrite violations file with cleaned data
    backup_file = log_dir / f"violations_{today}.jsonl.backup"
    log_file.rename(backup_file)
    print(f"\nBacked up original to: {backup_file}")
    
    with open(log_file, "w") as f:
        for v in cleaned_violations:
            f.write(json.dumps(v) + "\n")
    
    print(f"Rewrote cleaned violations to: {log_file}")
    
    # Generate insights
    threat_types = {}
    for v in cleaned_violations:
        t_type = v.get("threat_type", "unknown")
        threat_types[t_type] = threat_types.get(t_type, 0) + 1
    
    insights = {
        "timestamp": datetime.now().isoformat(),
        "threat_distribution": threat_types,
        "total_violations": sum(threat_types.values())
    }
    
    insights_file = log_dir / "insights.json"
    with open(insights_file, "w") as f:
        json.dump(insights, f, indent=2)
    
    print(f"\nUpdated insights:")
    print(f"  Total violations: {insights['total_violations']}")
    print(f"  Threat distribution:")
    for threat, count in sorted(threat_types.items(), key=lambda x: -x[1]):
        print(f"    - {threat}: {count}")
    
    print(f"\nInsights saved to: {insights_file}")

if __name__ == "__main__":
    main()
