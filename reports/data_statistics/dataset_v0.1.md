# PromptSec-Dataset v0.1 statistics

This report is generated deterministically from canonical records. P95 uses the nearest-rank method.

## Overview

- Total records: **627**
- Total spans: **288**
- Materialized after exact deduplication: **625**
- Dropped exact duplicates: **2**
- Semantic clusters: **399**
- Review queue: **627**

## Distributions

### Source

| Value | Records | Share |
|---|---:|---:|
| bipia | 250 | 39.87% |
| notinject | 339 | 54.07% |
| open_prompt_injection | 28 | 4.47% |
| promptinject | 10 | 1.59% |

### Language

| Value | Records | Share |
|---|---:|---:|
| en | 543 | 86.60% |
| und | 84 | 13.40% |

### Domain

| Value | Records | Share |
|---|---:|---:|
| Alphanumeric Substitution | 5 | 0.80% |
| Anagramming | 5 | 0.80% |
| Base Encoding | 5 | 0.80% |
| Blocking Internet Connection | 5 | 0.80% |
| Bringing Down Hosts and Servers (Denial of Service) | 5 | 0.80% |
| Business Intelligence | 5 | 0.80% |
| Clickbait | 5 | 0.80% |
| Common Queries | 126 | 20.10% |
| Compromising Computers | 5 | 0.80% |
| Content Creation | 5 | 0.80% |
| Conversational Agent | 5 | 0.80% |
| Cookie Theft | 5 | 0.80% |
| Corrupting an Operating System | 5 | 0.80% |
| Crippling Critical Infrastructures | 5 | 0.80% |
| Cryptocurrency Mining | 5 | 0.80% |
| Data Eavesdropping | 5 | 0.80% |
| Device and Driver Enumeration | 5 | 0.80% |
| Dumpster Diving | 5 | 0.80% |
| Emoji Substitution | 5 | 0.80% |
| Encrypting Documents and Demanding Ransom (Ransomware) | 5 | 0.80% |
| Entertainment | 5 | 0.80% |
| Environment Variable Analysis | 5 | 0.80% |
| Exploiting System Vulnerabilities | 5 | 0.80% |
| Homophonic Substitution | 5 | 0.80% |
| Information Dissemination | 5 | 0.80% |
| Information Retrieval | 5 | 0.80% |
| Instruction | 5 | 0.80% |
| Introduce System Fingerprinting | 5 | 0.80% |
| Keylogging | 5 | 0.80% |
| Language Translation | 10 | 1.59% |
| Learning and Tutoring | 5 | 0.80% |
| Malware Distribution | 5 | 0.80% |
| Marketing & Advertising | 5 | 0.80% |
| Memory Scanning | 5 | 0.80% |
| Misinformation & Propaganda | 5 | 0.80% |
| Misspelling Intentionally | 5 | 0.80% |
| Multilingual | 84 | 13.40% |
| Network Propagation | 5 | 0.80% |
| Persuasion | 5 | 0.80% |
| Programming Help | 5 | 0.80% |
| Research Assistance | 5 | 0.80% |
| Reverse Text | 5 | 0.80% |
| Scams & Fraud | 5 | 0.80% |
| Screen Scraping | 5 | 0.80% |
| Sending Out Spam Emails | 5 | 0.80% |
| Sentiment Analysis | 5 | 0.80% |
| Social Interaction | 5 | 0.80% |
| Space Removal & Grouping | 5 | 0.80% |
| Substitution Ciphers | 5 | 0.80% |
| Task Automation | 5 | 0.80% |
| Technique Queries | 87 | 13.88% |
| Traffic Analysis | 5 | 0.80% |
| Virtual Creation | 42 | 6.70% |
| duplicate_sentence_detection | 4 | 0.64% |
| goal_hikacking_attacks | 5 | 0.80% |
| grammar_correction | 4 | 0.64% |
| hate_detection | 4 | 0.64% |
| natural_language_inference | 4 | 0.64% |
| prompt_leaking_attacks | 5 | 0.80% |
| sentiment_analysis | 4 | 0.64% |
| spam_detection | 4 | 0.64% |
| summarization | 4 | 0.64% |

### Attack family

| Value | Records | Share |
|---|---:|---:|
| NONE | 339 | 54.07% |
| PROMPT_INJECTION | 288 | 45.93% |

### Attack objective

| Value | Records | Share |
|---|---:|---:|
| NONE | 339 | 54.07% |
| PROMPT_OR_POLICY_DISCLOSURE | 5 | 0.80% |
| TASK_HIJACKING | 283 | 45.14% |

### Delivery mode

| Value | Records | Share |
|---|---:|---:|
| DIRECT | 349 | 55.66% |
| INDIRECT | 278 | 44.34% |

### Annotation tier

| Value | Records | Share |
|---|---:|---:|
| DETERMINISTIC_MAPPING | 10 | 1.59% |
| HEURISTIC_MAPPING | 278 | 44.34% |
| UNANNOTATED | 339 | 54.07% |

## Mapping quality

| Indicator | Records | Rate |
|---|---:|---:|
| Gold source | 0 | 0.00% |
| Deterministic mapping | 10 | 1.59% |
| Heuristic mapping | 278 | 44.34% |
| Unannotated | 339 | 54.07% |
| Requires manual review | 617 | 98.41% |
| Review candidates | 627 | 100.00% |
| Review queue after threshold | 627 | 100.00% |
| Missing mapping confidence | 0 | 0.00% |

## UNKNOWN and UNDETERMINED fields

| Field | UNKNOWN | UNDETERMINED | Combined | Rate |
|---|---:|---:|---:|---:|
| annotations.authority_status | 339 | 0 | 339 | 54.07% |
| annotations.instruction_addressee | 339 | 0 | 339 | 54.07% |
| annotations.instruction_presence | 0 | 339 | 339 | 54.07% |
| annotations.instruction_presentation | 339 | 0 | 339 | 54.07% |
| annotations.protected_policy_alignment | 0 | 627 | 627 | 100.00% |
| annotations.user_goal_alignment | 0 | 339 | 339 | 54.07% |
| content.content_origin | 178 | 0 | 178 | 28.39% |
| content.delivery_mode | 0 | 0 | 0 | 0.00% |
| content.ingestion_path | 0 | 0 | 0 | 0.00% |
| content.source_integrity | 627 | 0 | 627 | 100.00% |
| content.source_role | 0 | 0 | 0 | 0.00% |

## Lengths

| Measure | Count | Min | Mean | Median | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Content characters | 627 | 11 | 116.810207 | 89.0 | 320 | 593 |
| Content UTF-8 bytes | 627 | 17 | 123.427432 | 93.0 | 320 | 593 |
| Spans per record | 627 | 0 | 0.45933 | 0.0 | 1 | 1 |
| Span length in characters | 288 | 17 | 151.496528 | 85.5 | 387 | 593 |

## Span types

| Value | Records | Share |
|---|---:|---:|
| INJECTION_PAYLOAD | 288 | 100.00% |
