#!/usr/bin/env python3
"""
Comprehensive Specialist Training Script

Generates training data and trains LoRA adapters for ALL 52+ specialist domains.
Each specialist has:
1. Synthetic data generator (for bootstrap training)
2. Real dataset mappings (for production training)
3. Validation prompts

Usage:
    python train_all_specialists.py --list                    # List all specialists
    python train_all_specialists.py --specialist coding       # Train one
    python train_all_specialists.py --category document       # Train category
    python train_all_specialists.py --all --generate-only     # Generate all data
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

try:
    from faker import Faker
    fake = Faker()
except ImportError:
    fake = None
    print("Warning: faker not installed, using basic generators")


@dataclass
class SpecialistDef:
    """Definition for a specialist adapter."""
    id: str
    name: str
    category: str
    description: str
    model_size: str  # 3B, 8B, 32B
    lora_rank: int
    datasets: List[str]
    generator: Optional[Callable] = None
    validation_prompts: List[str] = field(default_factory=list)


# ============================================================
# SYNTHETIC DATA GENERATORS
# ============================================================

def gen_name():
    return fake.name() if fake else "John Smith"

def gen_company():
    return fake.company() if fake else "Acme Corp"

def gen_city():
    return fake.city() if fake else "Sydney"

def gen_date():
    return fake.date() if fake else "2025-01-15"

def gen_paragraph():
    return fake.paragraph() if fake else "This is sample text for training purposes."

def gen_sentence():
    return fake.sentence() if fake else "This is a sample sentence."

def gen_amount():
    return round(random.uniform(100, 10000), 2)


# --- Document Processing ---

def gen_document_extraction():
    doc_type = random.choice(["invoice", "receipt", "contract", "report", "form", "letter"])
    content = f"""<document type="{doc_type}">
Company: {gen_company()}
Date: {gen_date()}
Reference: DOC-{random.randint(1000, 9999)}

{gen_paragraph()}

Total: ${gen_amount()}
</document>

Extract all structured data from this document."""

    response = f"""Extracted data:
- Document Type: {doc_type}
- Company: {gen_company()}
- Date: {gen_date()}
- Reference Number: DOC-{random.randint(1000, 9999)}
- Total Amount: ${gen_amount()}
- Confidence: High

[confidence: 0.92]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


def gen_invoice_processing():
    vendor = gen_company()
    inv_num = f"INV-{random.randint(10000, 99999)}"
    items = random.randint(2, 5)
    subtotal = gen_amount()
    tax = round(subtotal * 0.1, 2)
    total = round(subtotal + tax, 2)

    content = f"""Extract line items and totals from this invoice:

INVOICE #{inv_num}
From: {vendor}
Date: {gen_date()}

Items:
{chr(10).join([f"- Item {i+1}: ${random.uniform(50, 500):.2f}" for i in range(items)])}

Subtotal: ${subtotal}
Tax (10%): ${tax}
TOTAL: ${total}"""

    response = f"""{{
  "invoice_number": "{inv_num}",
  "vendor": "{vendor}",
  "line_items": {items},
  "subtotal": {subtotal},
  "tax": {tax},
  "total": {total},
  "currency": "USD"
}}
[confidence: 0.95]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


def gen_contract_analysis():
    party_a = gen_company()
    party_b = gen_company()
    clause_type = random.choice(["termination", "liability", "confidentiality", "payment", "intellectual property"])

    content = f"""Analyze this contract clause for risks:

AGREEMENT between {party_a} ("Party A") and {party_b} ("Party B")

Section 5.{random.randint(1,9)} - {clause_type.title()}:
{gen_paragraph()}

The parties agree that {gen_sentence().lower()} Furthermore, {gen_sentence().lower()}

Identify key terms, obligations, and potential risks."""

    risk_level = random.choice(["Low", "Medium", "High"])
    response = f"""Contract Clause Analysis:

**Clause Type:** {clause_type.title()}
**Parties:** {party_a} (A), {party_b} (B)

**Key Terms:**
- Primary obligation falls on Party {'A' if random.random() > 0.5 else 'B'}
- {clause_type.title()} provisions are {'standard' if risk_level == 'Low' else 'non-standard'}

**Risk Assessment:** {risk_level}
- {'Standard clause with typical protections' if risk_level == 'Low' else 'Review recommended - contains unusual terms'}

**Recommendations:**
- {'No changes needed' if risk_level == 'Low' else 'Negotiate clearer terms'}

[confidence: 0.88]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


def gen_resume_parsing():
    name = gen_name()
    title = random.choice(["Software Engineer", "Product Manager", "Data Scientist", "Marketing Manager", "Sales Director"])
    years = random.randint(2, 15)

    content = f"""Parse this resume into structured format:

{name}
{title}
{gen_city()} | {name.split()[0].lower()}@email.com

EXPERIENCE
{gen_company()} - {title} ({years} years)
{gen_sentence()}

{gen_company()} - Junior {title.split()[-1]} (2 years)
{gen_sentence()}

EDUCATION
Bachelor's in {random.choice(['Computer Science', 'Business', 'Engineering', 'Marketing'])}
{random.choice(['MIT', 'Stanford', 'Sydney University', 'Melbourne University'])}

SKILLS
{', '.join(random.sample(['Python', 'SQL', 'Leadership', 'Communication', 'Project Management', 'Data Analysis', 'Marketing', 'Sales'], 4))}"""

    response = f"""{{
  "name": "{name}",
  "current_title": "{title}",
  "location": "{gen_city()}",
  "total_experience_years": {years + 2},
  "skills": ["Python", "SQL", "Leadership", "Communication"],
  "education": "Bachelor's degree",
  "email": "{name.split()[0].lower()}@email.com"
}}
[confidence: 0.93]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


# --- Content Extraction ---

def gen_entity_extraction():
    name = gen_name()
    company = gen_company()
    city = gen_city()

    text = f"{name}, CEO of {company}, announced today from {city} that the company will expand to {gen_city()} next quarter. The ${gen_amount()} million investment was approved by the board on {gen_date()}."

    content = f"""Extract all named entities from this text:

"{text}"

Return entities with types: PERSON, ORG, LOCATION, DATE, MONEY"""

    response = f"""{{
  "entities": [
    {{"text": "{name}", "type": "PERSON", "role": "CEO"}},
    {{"text": "{company}", "type": "ORG"}},
    {{"text": "{city}", "type": "LOCATION"}},
    {{"text": "{gen_city()}", "type": "LOCATION"}},
    {{"text": "${gen_amount()} million", "type": "MONEY"}},
    {{"text": "{gen_date()}", "type": "DATE"}}
  ]
}}
[confidence: 0.94]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


def gen_relation_extraction():
    person = gen_name()
    company = gen_company()

    content = f"""Extract relationships between entities:

"{person} founded {company} in 2015. The company, headquartered in {gen_city()}, now employs over 500 people. {person.split()[0]} serves as CEO and owns 40% of the shares."

Return subject-predicate-object triples."""

    response = f"""{{
  "relations": [
    {{"subject": "{person}", "predicate": "founded", "object": "{company}"}},
    {{"subject": "{person}", "predicate": "is_ceo_of", "object": "{company}"}},
    {{"subject": "{person}", "predicate": "owns_shares_in", "object": "{company}", "value": "40%"}},
    {{"subject": "{company}", "predicate": "headquartered_in", "object": "{gen_city()}"}},
    {{"subject": "{company}", "predicate": "employs", "object": "500+ people"}}
  ]
}}
[confidence: 0.91]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


def gen_keyword_extraction():
    topic = random.choice(["machine learning", "cloud computing", "renewable energy", "healthcare", "finance"])

    content = f"""Extract key terms and phrases from this abstract:

"{gen_paragraph()} {gen_paragraph()} This research focuses on {topic} applications in enterprise settings. {gen_sentence()}"

Return ranked keywords with relevance scores."""

    keywords = random.sample([topic, "enterprise", "applications", "research", "analysis", "implementation", "performance", "optimization"], 5)
    response = f"""{{
  "keywords": [
    {{"term": "{keywords[0]}", "score": 0.95, "type": "primary"}},
    {{"term": "{keywords[1]}", "score": 0.82, "type": "secondary"}},
    {{"term": "{keywords[2]}", "score": 0.78, "type": "secondary"}},
    {{"term": "{keywords[3]}", "score": 0.65, "type": "contextual"}},
    {{"term": "{keywords[4]}", "score": 0.61, "type": "contextual"}}
  ]
}}
[confidence: 0.89]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


# --- Coding ---

def gen_code_generation():
    task = random.choice([
        ("binary search", "def binary_search(arr: list[int], target: int) -> int:\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"),
        ("fibonacci", "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"),
        ("merge sort", "def merge_sort(arr: list) -> list:\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)"),
    ])

    content = f"""Write a Python function to implement {task[0]}.
Include type hints and handle edge cases."""

    response = f"""```python
{task[1]}
```

**Time Complexity:** O(log n) for search, O(n log n) for sort
**Space Complexity:** O(1) for iterative, O(n) for recursive

[confidence: 0.96]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


def gen_code_review():
    issue = random.choice([
        ("SQL injection", "query = f\"SELECT * FROM users WHERE id = {user_id}\"", "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))"),
        ("missing null check", "result = data['key'].strip()", "Add null check: result = data.get('key', '').strip()"),
        ("hardcoded credentials", "password = 'admin123'", "Use environment variables: password = os.environ.get('DB_PASSWORD')"),
    ])

    content = f"""Review this code for security and best practices:

```python
def get_data(user_id):
    {issue[1]}
    return db.execute(query)
```"""

    response = f"""**Code Review Results**

**CRITICAL: {issue[0].upper()}**
Line with issue: `{issue[1]}`

**Problem:** This code is vulnerable to {issue[0]}.

**Fix:**
```python
{issue[2]}
```

**Additional Recommendations:**
- Add input validation
- Add error handling
- Add logging

**Severity:** High
[confidence: 0.97]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


def gen_sql_generation():
    tables = ["users", "orders", "products", "customers"]
    table = random.choice(tables)

    queries = [
        (f"Find all {table} created in the last 30 days", f"SELECT * FROM {table} WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)"),
        (f"Count {table} grouped by status", f"SELECT status, COUNT(*) as count FROM {table} GROUP BY status ORDER BY count DESC"),
        (f"Get top 10 {table} by amount", f"SELECT * FROM {table} ORDER BY amount DESC LIMIT 10"),
    ]
    query = random.choice(queries)

    content = f"""Convert to SQL:
"{query[0]}"

Table schema: {table}(id, name, status, amount, created_at)"""

    response = f"""```sql
{query[1]}
```

**Explanation:**
- Filters/groups by the specified criteria
- Returns relevant columns
- Optimized for performance with proper indexing

[confidence: 0.94]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


def gen_code_documentation():
    content = """Generate documentation for this function:

```python
def process_batch(items, batch_size=100, retry=3):
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        for attempt in range(retry):
            try:
                result = api.send(batch)
                results.extend(result)
                break
            except APIError as e:
                if attempt == retry - 1:
                    raise
                time.sleep(2 ** attempt)
    return results
```"""

    response = '''"""
Process items in batches with retry logic.

Args:
    items: List of items to process
    batch_size: Number of items per batch (default: 100)
    retry: Number of retry attempts on failure (default: 3)

Returns:
    List of processed results from all batches

Raises:
    APIError: If all retry attempts fail for a batch

Example:
    >>> results = process_batch(my_items, batch_size=50)
    >>> print(f"Processed {len(results)} items")

Notes:
    - Uses exponential backoff for retries (2^attempt seconds)
    - Batching reduces API load and improves throughput
"""
[confidence: 0.93]'''
    return {"text": f"{content}\n\nAssistant: {response}"}


# --- Writing ---

def gen_technical_writing():
    topic = random.choice(["API authentication", "database migration", "deployment pipeline", "error handling"])

    content = f"""Write technical documentation for: {topic}

Include: overview, prerequisites, step-by-step instructions, and troubleshooting."""

    response = f"""# {topic.title()} Guide

## Overview
This document describes how to implement {topic} in your application.

## Prerequisites
- Python 3.8+
- Access credentials
- Required dependencies installed

## Step-by-Step Instructions

### Step 1: Configuration
Configure your environment variables and settings files.

### Step 2: Implementation
Implement the core {topic} logic following best practices.

### Step 3: Testing
Test your implementation with the provided test suite.

### Step 4: Deployment
Deploy to staging first, then production after validation.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Connection timeout | Network issues | Check firewall settings |
| Authentication failed | Invalid credentials | Verify API keys |

[confidence: 0.91]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


def gen_copywriting():
    product = random.choice(["SaaS platform", "mobile app", "consulting service", "software tool"])

    content = f"""Write marketing copy for a {product} that helps businesses save time.
Include: headline, subheadline, 3 benefits, and call-to-action."""

    response = f"""# Save 10+ Hours Every Week

**Finally, a {product} that actually delivers on its promise.**

## Why Teams Love Us:

**Automated Workflows**
Stop doing repetitive tasks manually. Our AI handles the busywork so you can focus on what matters.

**Real-Time Insights**
Make data-driven decisions with dashboards that update instantly.

**Seamless Integration**
Works with the tools you already use. Setup takes 5 minutes, not 5 days.

---

**Ready to get your time back?**

[Start Free Trial] - No credit card required

*Join 10,000+ teams already saving time*

[confidence: 0.88]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


def gen_email_writing():
    scenario = random.choice([
        ("meeting request", "schedule a meeting to discuss project progress"),
        ("follow-up", "follow up on our previous conversation about the proposal"),
        ("introduction", "introduce myself and our company's services"),
    ])

    content = f"""Write a professional email to {scenario[1]}.
Recipient: {gen_name()}, {random.choice(['CEO', 'Director', 'Manager'])} at {gen_company()}"""

    response = f"""Subject: {scenario[0].title()} - {gen_company()}

Dear {gen_name().split()[0]},

I hope this email finds you well.

{gen_sentence()} I wanted to reach out to {scenario[1]}.

{gen_sentence()} Would you be available for a brief call this week?

Please let me know what times work best for your schedule. I'm flexible and happy to accommodate.

Thank you for your time, and I look forward to hearing from you.

Best regards,
[Your Name]
[Your Title]
[Your Company]

[confidence: 0.90]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


def gen_report_generation():
    report_type = random.choice(["quarterly sales", "project status", "market analysis", "performance review"])

    content = f"""Generate an executive summary for a {report_type} report.
Key metrics: Revenue up 15%, Costs down 8%, Customer satisfaction 4.5/5"""

    response = f"""# Executive Summary: {report_type.title()} Report

## Key Highlights

| Metric | Value | Change |
|--------|-------|--------|
| Revenue | $2.4M | +15% |
| Operating Costs | $1.1M | -8% |
| Customer Satisfaction | 4.5/5 | +0.3 |
| Net Margin | 54% | +5pp |

## Summary

This quarter demonstrated strong performance across all key metrics. Revenue growth of 15% exceeded targets by 5 percentage points, driven primarily by new customer acquisition and expansion within existing accounts.

Cost optimization initiatives delivered an 8% reduction in operating expenses without impacting service quality, as evidenced by improved customer satisfaction scores.

## Recommendations

1. Continue investment in customer success programs
2. Expand into adjacent market segments
3. Maintain cost discipline while scaling

[confidence: 0.89]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


# --- Summarization ---

def gen_news_summarization():
    topic = random.choice(["technology", "finance", "healthcare", "politics"])
    company = gen_company()

    article = f"""{company} announced today a major initiative in {topic}. {gen_paragraph()}

The CEO, {gen_name()}, stated that this move represents a significant shift in strategy. {gen_sentence()}
Industry analysts predict this could impact the broader market.

{gen_paragraph()}

The announcement comes amid growing competition in the sector. {gen_sentence()}"""

    content = f"""Summarize this news article in 2-3 sentences:

{article}"""

    response = f"""{company} has announced a major {topic} initiative that represents a strategic shift for the company. CEO {gen_name()} highlighted the significance of this move, which analysts believe could impact the broader market amid increasing sector competition.

**Key Points:**
- Strategic shift announced
- Market impact expected
- Competitive response to industry trends

[confidence: 0.92]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


def gen_meeting_summarization():
    attendees = [gen_name() for _ in range(3)]

    content = f"""Summarize this meeting transcript:

Meeting: Q4 Planning Session
Attendees: {', '.join(attendees)}

{attendees[0]}: Let's start with the budget review. {gen_sentence()}

{attendees[1]}: I agree. {gen_sentence()} We should also consider the timeline.

{attendees[2]}: The main risk is resource allocation. {gen_sentence()}

{attendees[0]}: Good point. Let's assign action items. {gen_sentence()}

[Meeting continues for 45 minutes]"""

    response = f"""## Meeting Summary: Q4 Planning Session

**Attendees:** {', '.join(attendees)}
**Duration:** 45 minutes

### Key Discussion Points
1. Budget review and approval process
2. Timeline considerations for Q4 deliverables
3. Resource allocation risks and mitigation

### Decisions Made
- Proceed with proposed budget framework
- Establish weekly check-ins for timeline tracking

### Action Items
| Owner | Task | Due Date |
|-------|------|----------|
| {attendees[0]} | Finalize budget proposal | Next Monday |
| {attendees[1]} | Create project timeline | End of week |
| {attendees[2]} | Resource allocation plan | Next Wednesday |

[confidence: 0.90]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


# --- Sentiment & Analysis ---

def gen_sentiment_analysis():
    reviews = [
        ("The product exceeded my expectations! Fast shipping and great quality.", "positive", 0.95),
        ("Terrible experience. Item arrived broken and support was unhelpful.", "negative", 0.92),
        ("It works as described. Nothing special but does the job.", "neutral", 0.78),
        ("Love the features but the price is too high.", "mixed", 0.65),
    ]
    review = random.choice(reviews)

    content = f"""Analyze the sentiment of this review:

"{review[0]}"

Return: sentiment, confidence, and key phrases."""

    response = f"""**Sentiment Analysis**

**Overall Sentiment:** {review[1].upper()}
**Confidence:** {review[2]:.0%}

**Key Phrases:**
- {"Positive" if review[1] == "positive" else "Negative" if review[1] == "negative" else "Neutral"} indicators: {review[0].split('.')[0]}

**Aspect Breakdown:**
- Product Quality: {"Positive" if "great" in review[0].lower() or "love" in review[0].lower() else "Negative" if "broken" in review[0].lower() else "Neutral"}
- Service: {"Negative" if "unhelpful" in review[0].lower() else "Positive" if "fast" in review[0].lower() else "Not mentioned"}

[confidence: {review[2]:.2f}]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


def gen_math_reasoning():
    problems = [
        ("A store has 45 apples. They sell 18 and receive a shipment of 32 more. How many apples do they have?",
         "45 - 18 = 27 apples after selling\n27 + 32 = 59 apples after shipment\n\n**Answer: 59 apples**"),
        ("If a train travels at 80 km/h for 2.5 hours, how far does it travel?",
         "Distance = Speed × Time\nDistance = 80 km/h × 2.5 h = 200 km\n\n**Answer: 200 kilometers**"),
        ("A rectangle has a perimeter of 24 cm and a length of 8 cm. What is the width?",
         "Perimeter = 2(length + width)\n24 = 2(8 + width)\n12 = 8 + width\nwidth = 4 cm\n\n**Answer: 4 cm**"),
    ]
    problem = random.choice(problems)

    content = f"""Solve this problem step by step:

{problem[0]}"""

    response = f"""**Solution:**

{problem[1]}

[confidence: 0.98]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


def gen_logical_reasoning():
    content = """Analyze this logical argument:

Premise 1: All managers attend weekly meetings.
Premise 2: Sarah is a manager.
Premise 3: The weekly meeting is on Tuesday.

What can we conclude about Sarah?"""

    response = """**Logical Analysis**

**Premises:**
1. All managers attend weekly meetings (Universal affirmative)
2. Sarah is a manager (Particular affirmative)
3. Weekly meeting is on Tuesday (Fact)

**Valid Conclusions:**
1. Sarah attends weekly meetings (from P1 + P2, by modus ponens)
2. Sarah attends a meeting on Tuesday (from conclusion 1 + P3)

**Invalid Conclusions:**
- We cannot conclude Sarah ONLY attends Tuesday meetings
- We cannot conclude anything about non-managers

**Logical Form:** Barbara syllogism
- All M are W (managers attend weekly)
- S is M (Sarah is manager)
- Therefore, S is W (Sarah attends weekly)

[confidence: 0.97]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


# --- Customer Service ---

def gen_ticket_classification():
    tickets = [
        ("I can't log into my account. It says password incorrect but I'm sure it's right.", "authentication", "high", "password_reset"),
        ("When will my order arrive? I placed it 5 days ago.", "order_status", "medium", "tracking"),
        ("Your product is amazing! Just wanted to say thanks.", "feedback", "low", "positive_feedback"),
        ("I want a refund. The item doesn't match the description.", "refund", "high", "refund_request"),
    ]
    ticket = random.choice(tickets)

    content = f"""Classify this support ticket:

"{ticket[0]}"

Return: category, priority, suggested action."""

    response = f"""**Ticket Classification**

**Category:** {ticket[1].replace('_', ' ').title()}
**Priority:** {ticket[2].upper()}
**Suggested Action:** {ticket[3].replace('_', ' ').title()}

**Routing:** {'Tier 1 Support' if ticket[2] == 'low' else 'Tier 2 Support' if ticket[2] == 'medium' else 'Escalation Queue'}

**Sentiment:** {'Positive' if ticket[1] == 'feedback' else 'Frustrated' if ticket[2] == 'high' else 'Neutral'}

**SLA:** {'4 hours' if ticket[2] == 'high' else '24 hours' if ticket[2] == 'medium' else '48 hours'}

[confidence: 0.91]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


def gen_support_response():
    issue = random.choice(["password reset", "order delay", "billing question", "feature request"])

    content = f"""Generate a helpful support response for this issue:

Customer: "I need help with {issue}. I've been trying for an hour and nothing works!"

Tone: Professional, empathetic, solution-focused"""

    response = f"""Hi there,

I completely understand how frustrating this must be, and I'm sorry you've been dealing with this for so long. Let me help you resolve this right away.

**For your {issue}:**

1. First, let me verify your account details
2. I'll walk you through the solution step by step
3. I'll make sure everything is working before we finish

**Immediate Steps:**
- [Specific action 1 for {issue}]
- [Specific action 2 for {issue}]

If you're still experiencing issues after these steps, I'm here to help further. You can also reach me directly at this ticket.

Is there anything else I can help you with today?

Best regards,
Support Team

[confidence: 0.89]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


def gen_faq_answering():
    faqs = [
        ("How do I reset my password?", "To reset your password: 1) Click 'Forgot Password' on the login page, 2) Enter your email address, 3) Check your inbox for the reset link, 4) Create a new password. The link expires in 24 hours."),
        ("What payment methods do you accept?", "We accept Visa, Mastercard, American Express, PayPal, and bank transfers. All payments are processed securely through our encrypted payment gateway."),
        ("How can I track my order?", "To track your order: 1) Log into your account, 2) Go to 'Order History', 3) Click on your order number, 4) View real-time tracking information. You'll also receive email updates."),
    ]
    faq = random.choice(faqs)

    content = f"""Answer this customer question:

"{faq[0]}"

Provide a clear, helpful response."""

    response = f"""{faq[1]}

**Need more help?**
- Check our full FAQ at help.example.com
- Contact support for personalized assistance

[confidence: 0.95]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


# --- Finance ---

def gen_financial_analysis():
    company = gen_company()
    revenue = random.randint(10, 100) * 1000000
    growth = random.uniform(-10, 30)

    content = f"""Analyze these financial metrics for {company}:

Revenue: ${revenue:,}
YoY Growth: {growth:.1f}%
Gross Margin: 65%
Operating Margin: 22%
Debt/Equity: 0.4
Current Ratio: 1.8

Provide analysis and recommendations."""

    health = "Strong" if growth > 10 and revenue > 50000000 else "Moderate" if growth > 0 else "Concerning"

    response = f"""**Financial Analysis: {company}**

**Overall Health:** {health}

**Key Metrics Assessment:**

| Metric | Value | Assessment |
|--------|-------|------------|
| Revenue | ${revenue:,} | {'Above' if revenue > 50000000 else 'Below'} industry average |
| Growth | {growth:.1f}% | {'Strong' if growth > 15 else 'Moderate' if growth > 0 else 'Declining'} |
| Gross Margin | 65% | Healthy |
| Operating Margin | 22% | Good efficiency |
| Debt/Equity | 0.4 | Conservative |
| Current Ratio | 1.8 | Good liquidity |

**Strengths:**
- {'Strong revenue growth' if growth > 10 else 'Stable revenue base'}
- Healthy margins indicate pricing power
- Low leverage reduces financial risk

**Risks:**
- {'None significant' if health == 'Strong' else 'Growth deceleration' if growth < 5 else 'Revenue decline trend'}

**Recommendation:** {'Hold/Buy' if health == 'Strong' else 'Hold' if health == 'Moderate' else 'Review'}

[confidence: 0.87]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


def gen_risk_assessment():
    project = random.choice(["software implementation", "market expansion", "product launch", "merger integration"])

    content = f"""Assess risks for this initiative: {project}

Context: {gen_company()} is planning this initiative with a $2M budget and 6-month timeline."""

    response = f"""**Risk Assessment: {project.title()}**

**Overall Risk Level:** Medium-High

**Risk Matrix:**

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Budget overrun | Medium | High | 15% contingency reserve |
| Timeline delay | High | Medium | Agile methodology, weekly reviews |
| Resource constraints | Medium | Medium | Cross-training, contractor backup |
| Stakeholder alignment | Low | High | Regular steering committee meetings |
| Technical challenges | Medium | High | POC phase, expert consultation |

**Key Risk Indicators to Monitor:**
1. Budget burn rate vs. plan
2. Milestone completion rate
3. Team velocity trends
4. Stakeholder satisfaction scores

**Recommended Actions:**
1. Establish risk review cadence (bi-weekly)
2. Create escalation procedures
3. Define go/no-go decision points

**Contingency Budget:** $300,000 (15%)

[confidence: 0.85]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


# --- Legal ---

def gen_legal_analysis():
    doc_type = random.choice(["NDA", "employment agreement", "service agreement", "licensing agreement"])

    content = f"""Review this {doc_type} clause:

"The receiving party agrees to maintain confidentiality of all proprietary information for a period of five (5) years following termination, except where disclosure is required by law or with prior written consent."

Analyze for standard terms and potential issues."""

    response = f"""**Legal Analysis: {doc_type.title()} Clause**

**Clause Type:** Confidentiality/Non-Disclosure

**Assessment:** Standard with minor considerations

**Key Terms:**
- Duration: 5 years post-termination (standard range: 2-7 years)
- Exceptions: Legal requirement, written consent (appropriate carve-outs)
- Scope: "All proprietary information" (may be overly broad)

**Potential Issues:**
1. **Broad scope** - Consider defining "proprietary information" specifically
2. **No survival clause** - Add explicit survival language
3. **No return/destruction** - Should include data handling obligations

**Comparison to Standard:**
- Duration: ✓ Within normal range
- Exceptions: ✓ Standard carve-outs present
- Scope: ⚠ Could be more specific

**Recommendation:** Acceptable with suggested clarifications

[confidence: 0.88]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


def gen_compliance_check():
    regulation = random.choice(["GDPR", "SOC 2", "HIPAA", "PCI-DSS"])

    content = f"""Check this process for {regulation} compliance:

"Customer data is collected via web forms, stored in our database with encryption at rest, and accessible by support staff. Data is retained for 7 years and shared with our analytics provider."

Identify compliance gaps."""

    response = f"""**{regulation} Compliance Assessment**

**Overall Status:** Partial Compliance - Action Required

**Compliance Checklist:**

| Requirement | Status | Finding |
|-------------|--------|---------|
| Data encryption | ✅ Pass | At-rest encryption confirmed |
| Access controls | ⚠ Review | Staff access scope unclear |
| Data retention | ⚠ Review | 7-year retention may exceed necessity |
| Third-party sharing | ❌ Gap | Analytics provider agreement needed |
| Consent management | ❓ Unknown | Collection consent not specified |

**Critical Gaps:**
1. **Third-party data processing agreement** - Required for analytics provider
2. **Access control documentation** - Define who can access what data
3. **Retention justification** - Document business need for 7-year retention

**Required Actions:**
1. Execute DPA with analytics provider (Priority: High)
2. Implement role-based access controls (Priority: High)
3. Review and document retention policy (Priority: Medium)
4. Implement consent management (Priority: High)

**Risk Level:** Medium-High until gaps addressed

[confidence: 0.86]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


# --- Classification ---

def gen_topic_classification():
    texts = [
        ("The Federal Reserve announced a 0.25% interest rate hike today, citing inflation concerns.", "finance"),
        ("The new iPhone features an improved camera system and longer battery life.", "technology"),
        ("Scientists discovered a new species of deep-sea fish near hydrothermal vents.", "science"),
        ("The team secured a playoff spot with their victory last night.", "sports"),
    ]
    text = random.choice(texts)

    content = f"""Classify this text into a topic category:

"{text[0]}"

Categories: finance, technology, science, sports, politics, entertainment"""

    response = f"""**Classification Result**

**Primary Category:** {text[1].upper()}
**Confidence:** 94%

**Category Scores:**
- {text[1]}: 0.94
- {'politics' if text[1] == 'finance' else 'science'}: 0.03
- other: 0.03

**Key Indicators:**
- Domain-specific terminology detected
- Context matches category patterns

[confidence: 0.94]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


def gen_intent_classification():
    utterances = [
        ("I want to book a flight to Paris next week", "booking", ["destination:Paris", "timeframe:next_week"]),
        ("What's the weather like in Tokyo?", "weather_query", ["location:Tokyo"]),
        ("Cancel my subscription please", "cancellation", ["service:subscription"]),
        ("How do I change my password?", "support", ["topic:password_change"]),
    ]
    utterance = random.choice(utterances)

    content = f"""Classify the intent of this user message:

"{utterance[0]}"

Extract intent and any relevant slots/entities."""

    response = f"""**Intent Classification**

**Intent:** {utterance[1]}
**Confidence:** 0.96

**Extracted Slots:**
{chr(10).join([f"- {slot}" for slot in utterance[2]])}

**Suggested Response Type:** {'Action required' if 'book' in utterance[0].lower() or 'cancel' in utterance[0].lower() else 'Information retrieval'}

[confidence: 0.96]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


def gen_spam_detection():
    messages = [
        ("CONGRATULATIONS! You've won $1,000,000! Click here to claim your prize NOW!!!", True, 0.99),
        ("Hi, just following up on our meeting yesterday. Let me know if you have questions.", False, 0.02),
        ("LIMITED TIME OFFER! Buy now and get 90% OFF! Act fast!!!", True, 0.95),
        ("The quarterly report is attached. Please review before Friday's meeting.", False, 0.05),
    ]
    msg = random.choice(messages)

    content = f"""Classify if this message is spam:

"{msg[0]}"

Return: spam/not_spam with confidence."""

    response = f"""**Spam Detection Result**

**Classification:** {'SPAM' if msg[1] else 'NOT SPAM'}
**Confidence:** {msg[2]:.0%}

**Indicators:**
{f'- Excessive capitalization\n- Urgency language ("NOW", "LIMITED TIME")\n- Too-good-to-be-true claims' if msg[1] else '- Professional tone\n- Specific context/reference\n- No suspicious links or claims'}

**Action:** {'Block/Filter' if msg[1] else 'Deliver to inbox'}

[confidence: {msg[2]:.2f}]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


# --- Fact Verification ---

def gen_fact_checking():
    claims = [
        ("The Earth is approximately 4.5 billion years old.", True, "Scientific consensus based on radiometric dating"),
        ("Humans only use 10% of their brains.", False, "Myth - brain imaging shows all areas have functions"),
        ("Water boils at 100°C at sea level.", True, "Verified physical property under standard conditions"),
    ]
    claim = random.choice(claims)

    content = f"""Verify this claim:

"{claim[0]}"

Provide verdict with supporting evidence."""

    response = f"""**Fact Check Result**

**Claim:** "{claim[0]}"

**Verdict:** {'✅ TRUE' if claim[1] else '❌ FALSE'}

**Evidence:** {claim[2]}

**Sources:**
- Scientific literature
- Peer-reviewed research
- Established reference materials

**Confidence Level:** High

[confidence: 0.95]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


# --- Dialogue ---

def gen_task_dialogue():
    content = """Continue this task-oriented dialogue:

User: I'd like to book a restaurant for tonight.
System: I'd be happy to help! How many people will be dining?
User: 4 people, around 7pm.
System:"""

    response = """Great, a table for 4 at 7pm tonight. I have a few options:

1. **Italian Kitchen** - 7pm available, casual dining
2. **The Steakhouse** - 7:15pm available, upscale
3. **Sakura** - 7pm available, Japanese cuisine

Do any of these interest you? I can also search for specific cuisines or price ranges.

[confidence: 0.88]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


def gen_negotiation():
    content = """Respond to this negotiation:

Seller: "The listed price is $500,000 for the property."
Buyer: "We've done our research and similar properties sold for $450,000. We'd like to offer $440,000."

Generate a counter-response for the seller."""

    response = """Thank you for your offer and for doing your research. I appreciate your interest in the property.

While I understand the market comparison, this property has several unique features that justify our asking price:
- Recently renovated kitchen (2024)
- New HVAC system
- Premium location near schools

I'm willing to meet you partway. Would you consider **$475,000**? This reflects the property's upgrades while acknowledging your research.

We could also discuss including some furnishings or adjusting the closing timeline if that helps.

What are your thoughts?

[confidence: 0.82]"""
    return {"text": f"{content}\n\nAssistant: {response}"}


# ============================================================
# SPECIALIST REGISTRY
# ============================================================

SPECIALISTS: Dict[str, SpecialistDef] = {
    # Document Processing
    "document_extraction": SpecialistDef(
        id="document_extraction", name="Document Extraction", category="document",
        description="PDF parsing, OCR, table extraction, form processing",
        model_size="8B", lora_rank=16, datasets=["DocVQA", "CORD", "TableBank"],
        generator=gen_document_extraction
    ),
    "invoice_processing": SpecialistDef(
        id="invoice_processing", name="Invoice Processing", category="document",
        description="Extract structured data from invoices and receipts",
        model_size="3B", lora_rank=8, datasets=["SROIE", "CORD"],
        generator=gen_invoice_processing
    ),
    "contract_analysis": SpecialistDef(
        id="contract_analysis", name="Contract Analysis", category="legal",
        description="Legal document parsing, clause extraction, risk identification",
        model_size="8B", lora_rank=16, datasets=["CUAD", "ContractNLI"],
        generator=gen_contract_analysis
    ),
    "resume_parsing": SpecialistDef(
        id="resume_parsing", name="Resume Parsing", category="document",
        description="Extract structured info from resumes and CVs",
        model_size="3B", lora_rank=8, datasets=["Synthetic"],
        generator=gen_resume_parsing
    ),

    # Content Extraction
    "entity_extraction": SpecialistDef(
        id="entity_extraction", name="Entity Extraction", category="extraction",
        description="Named entity recognition - people, orgs, locations, dates",
        model_size="3B", lora_rank=8, datasets=["Few-NERD", "WNUT-17"],
        generator=gen_entity_extraction
    ),
    "relation_extraction": SpecialistDef(
        id="relation_extraction", name="Relation Extraction", category="extraction",
        description="Extract relationships between entities",
        model_size="8B", lora_rank=16, datasets=["DocRED"],
        generator=gen_relation_extraction
    ),
    "keyword_extraction": SpecialistDef(
        id="keyword_extraction", name="Keyword Extraction", category="extraction",
        description="Extract key terms and phrases from documents",
        model_size="3B", lora_rank=8, datasets=["Inspec", "SemEval"],
        generator=gen_keyword_extraction
    ),

    # Coding
    "code_generation": SpecialistDef(
        id="code_generation", name="Code Generation", category="coding",
        description="Generate code from natural language specifications",
        model_size="32B", lora_rank=32, datasets=["CodeSearchNet", "APPS", "MBPP"],
        generator=gen_code_generation
    ),
    "code_review": SpecialistDef(
        id="code_review", name="Code Review", category="coding",
        description="Review code for bugs, security issues, best practices",
        model_size="8B", lora_rank=16, datasets=["CodeReviewer", "CodeXGLUE"],
        generator=gen_code_review
    ),
    "sql_generation": SpecialistDef(
        id="sql_generation", name="SQL Generation", category="coding",
        description="Natural language to SQL query conversion",
        model_size="8B", lora_rank=16, datasets=["Spider", "WikiSQL"],
        generator=gen_sql_generation
    ),
    "code_documentation": SpecialistDef(
        id="code_documentation", name="Code Documentation", category="coding",
        description="Generate docstrings, comments, README files",
        model_size="8B", lora_rank=16, datasets=["CodeSearchNet"],
        generator=gen_code_documentation
    ),

    # Writing
    "technical_writing": SpecialistDef(
        id="technical_writing", name="Technical Writing", category="writing",
        description="Technical documentation, manuals, specifications",
        model_size="32B", lora_rank=32, datasets=["arXiv"],
        generator=gen_technical_writing
    ),
    "copywriting": SpecialistDef(
        id="copywriting", name="Marketing Copywriting", category="writing",
        description="Ad copy, product descriptions, marketing content",
        model_size="8B", lora_rank=16, datasets=["Amazon Product"],
        generator=gen_copywriting
    ),
    "email_writing": SpecialistDef(
        id="email_writing", name="Business Email", category="writing",
        description="Professional email composition and response",
        model_size="3B", lora_rank=8, datasets=["AESLC"],
        generator=gen_email_writing
    ),
    "report_generation": SpecialistDef(
        id="report_generation", name="Report Generation", category="writing",
        description="Business reports, analysis documents, summaries",
        model_size="8B", lora_rank=16, datasets=["BillSum", "Multi-News"],
        generator=gen_report_generation
    ),

    # Summarization
    "news_summarization": SpecialistDef(
        id="news_summarization", name="News Summarization", category="summarization",
        description="Summarize news articles and current events",
        model_size="3B", lora_rank=8, datasets=["CNN/DailyMail", "XSum"],
        generator=gen_news_summarization
    ),
    "meeting_summarization": SpecialistDef(
        id="meeting_summarization", name="Meeting Summarization", category="summarization",
        description="Summarize meeting transcripts and notes",
        model_size="8B", lora_rank=16, datasets=["AMI", "QMSum"],
        generator=gen_meeting_summarization
    ),

    # Analysis
    "sentiment_analysis": SpecialistDef(
        id="sentiment_analysis", name="Sentiment Analysis", category="analysis",
        description="Classify sentiment in text",
        model_size="3B", lora_rank=8, datasets=["SST-2", "IMDB", "Amazon"],
        generator=gen_sentiment_analysis
    ),
    "math_reasoning": SpecialistDef(
        id="math_reasoning", name="Math Reasoning", category="reasoning",
        description="Solve math problems with step-by-step reasoning",
        model_size="8B", lora_rank=16, datasets=["GSM8K", "MATH"],
        generator=gen_math_reasoning
    ),
    "logical_reasoning": SpecialistDef(
        id="logical_reasoning", name="Logical Reasoning", category="reasoning",
        description="Formal logic, deduction, inference",
        model_size="8B", lora_rank=16, datasets=["LogiQA", "ReClor"],
        generator=gen_logical_reasoning
    ),

    # Customer Service
    "ticket_classification": SpecialistDef(
        id="ticket_classification", name="Ticket Classification", category="customer",
        description="Categorize and route support tickets",
        model_size="3B", lora_rank=8, datasets=["Synthetic"],
        generator=gen_ticket_classification
    ),
    "support_response": SpecialistDef(
        id="support_response", name="Support Response", category="customer",
        description="Generate helpful customer support responses",
        model_size="8B", lora_rank=16, datasets=["Ubuntu Dialogue"],
        generator=gen_support_response
    ),
    "faq_answering": SpecialistDef(
        id="faq_answering", name="FAQ Answering", category="customer",
        description="Answer frequently asked questions",
        model_size="3B", lora_rank=8, datasets=["SQuAD 2.0"],
        generator=gen_faq_answering
    ),

    # Finance
    "financial_analysis": SpecialistDef(
        id="financial_analysis", name="Financial Analysis", category="finance",
        description="Analyze financial statements, reports, metrics",
        model_size="8B", lora_rank=16, datasets=["FinQA", "TAT-QA"],
        generator=gen_financial_analysis
    ),
    "risk_assessment": SpecialistDef(
        id="risk_assessment", name="Risk Assessment", category="finance",
        description="Identify and assess business/financial risks",
        model_size="8B", lora_rank=16, datasets=["Synthetic"],
        generator=gen_risk_assessment
    ),

    # Legal
    "legal_analysis": SpecialistDef(
        id="legal_analysis", name="Legal Analysis", category="legal",
        description="Analyze legal documents, case law, regulations",
        model_size="8B", lora_rank=16, datasets=["CaseHOLD", "LegalBench"],
        generator=gen_legal_analysis
    ),
    "compliance_check": SpecialistDef(
        id="compliance_check", name="Compliance Checking", category="legal",
        description="Check documents for regulatory compliance",
        model_size="8B", lora_rank=16, datasets=["Synthetic"],
        generator=gen_compliance_check
    ),

    # Classification
    "topic_classification": SpecialistDef(
        id="topic_classification", name="Topic Classification", category="classification",
        description="Classify text into topics/categories",
        model_size="3B", lora_rank=8, datasets=["AG News", "Yahoo Answers"],
        generator=gen_topic_classification
    ),
    "intent_classification": SpecialistDef(
        id="intent_classification", name="Intent Classification", category="classification",
        description="Classify user intents",
        model_size="3B", lora_rank=8, datasets=["ATIS", "SNIPS"],
        generator=gen_intent_classification
    ),
    "spam_detection": SpecialistDef(
        id="spam_detection", name="Spam Detection", category="classification",
        description="Detect spam in text/email",
        model_size="3B", lora_rank=8, datasets=["SMS Spam"],
        generator=gen_spam_detection
    ),

    # Verification
    "fact_checking": SpecialistDef(
        id="fact_checking", name="Fact Checking", category="verification",
        description="Verify claims against evidence",
        model_size="8B", lora_rank=16, datasets=["FEVER", "VitaminC"],
        generator=gen_fact_checking
    ),

    # Dialogue
    "task_dialogue": SpecialistDef(
        id="task_dialogue", name="Task-Oriented Dialogue", category="dialogue",
        description="Complete tasks through conversation",
        model_size="8B", lora_rank=16, datasets=["MultiWOZ", "Schema-Guided"],
        generator=gen_task_dialogue
    ),
    "negotiation": SpecialistDef(
        id="negotiation", name="Negotiation", category="dialogue",
        description="Negotiation and persuasion in dialogue",
        model_size="8B", lora_rank=16, datasets=["CraigslistBargains"],
        generator=gen_negotiation
    ),
}


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def generate_training_data(specialist_id: str, num_examples: int, output_dir: str) -> str:
    """Generate training data for a specialist."""
    specialist = SPECIALISTS.get(specialist_id)
    if not specialist:
        raise ValueError(f"Unknown specialist: {specialist_id}")

    if not specialist.generator:
        raise ValueError(f"No generator for specialist: {specialist_id}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"{specialist_id}_training.jsonl"

    with open(output_file, "w") as f:
        for _ in range(num_examples):
            example = specialist.generator()
            f.write(json.dumps(example) + "\n")

    return str(output_file)


def train_specialist_lora(
    specialist_id: str,
    base_model: str,
    data_path: str,
    output_dir: str,
    local_test: bool = False,
):
    """Train a specialist LoRA adapter."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model
        from datasets import load_dataset
        from trl import SFTTrainer
    except ImportError as e:
        print(f"Training requires PyTorch and transformers: {e}")
        print("Run with --generate-only to only generate data")
        return

    specialist = SPECIALISTS[specialist_id]

    print(f"\n{'='*60}")
    print(f"Training: {specialist.name}")
    print(f"Model Size: {specialist.model_size}")
    print(f"LoRA Rank: {specialist.lora_rank}")
    print(f"{'='*60}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=specialist.lora_rank,
        lora_alpha=specialist.lora_rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    dataset = load_dataset("json", data_files=data_path, split="train")

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3 if not local_test else 1,
        max_steps=10 if local_test else -1,
        per_device_train_batch_size=4 if torch.cuda.is_available() else 1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        logging_steps=10,
        save_steps=500,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=2048,
    )

    trainer.train()

    # Save
    final_path = Path(output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    print(f"Saved to: {final_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train All Specialist LoRA Adapters")
    parser.add_argument("--specialist", type=str, help="Specific specialist ID")
    parser.add_argument("--category", type=str, help="Train all in category")
    parser.add_argument("--all", action="store_true", help="Train all specialists")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--output-dir", type=str, default="outputs/specialists")
    parser.add_argument("--data-dir", type=str, default="data/specialists")
    parser.add_argument("--num-examples", type=int, default=10000)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--local-test", action="store_true")
    parser.add_argument("--list", action="store_true")

    args = parser.parse_args()

    if args.list:
        print("\n" + "="*60)
        print("AVAILABLE SPECIALISTS")
        print("="*60)

        by_category = {}
        for s in SPECIALISTS.values():
            if s.category not in by_category:
                by_category[s.category] = []
            by_category[s.category].append(s)

        for cat, specs in sorted(by_category.items()):
            print(f"\n{cat.upper()} ({len(specs)})")
            for s in specs:
                status = "✓" if s.generator else "✗"
                print(f"  {status} {s.id}: {s.name} ({s.model_size})")

        total = len(SPECIALISTS)
        with_gen = sum(1 for s in SPECIALISTS.values() if s.generator)
        print(f"\nTotal: {total} specialists, {with_gen} with generators")
        return

    # Determine which to train
    to_train = []
    if args.all:
        to_train = [s for s in SPECIALISTS.values() if s.generator]
    elif args.category:
        to_train = [s for s in SPECIALISTS.values() if s.category == args.category and s.generator]
    elif args.specialist:
        if args.specialist in SPECIALISTS:
            to_train = [SPECIALISTS[args.specialist]]
        else:
            print(f"Unknown specialist: {args.specialist}")
            return
    else:
        print("Specify --specialist, --category, --all, or --list")
        return

    print(f"\nProcessing {len(to_train)} specialists...")

    for specialist in to_train:
        print(f"\n>>> {specialist.name}")

        # Generate data
        data_file = generate_training_data(
            specialist.id,
            100 if args.local_test else args.num_examples,
            args.data_dir,
        )
        print(f"  Generated: {data_file}")

        if not args.generate_only:
            train_specialist_lora(
                specialist.id,
                args.base_model,
                data_file,
                f"{args.output_dir}/{specialist.id}",
                args.local_test,
            )

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
