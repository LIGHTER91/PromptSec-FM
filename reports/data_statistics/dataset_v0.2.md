# promptsec-dataset-v0.2 statistics

This report is generated deterministically from canonical records. P95 uses the nearest-rank method.

## Overview

- Total records: **3684**
- Total spans: **5453**
- Materialized after exact deduplication: **3682**
- Dropped exact duplicates: **2**
- Semantic clusters: **491**
- Review queue: **3684**

## Distributions

### Source

| Value | Records | Share |
|---|---:|---:|
| agentdojo | 949 | 25.76% |
| bipia | 250 | 6.79% |
| injecagent | 2108 | 57.22% |
| notinject | 339 | 9.20% |
| open_prompt_injection | 28 | 0.76% |
| promptinject | 10 | 0.27% |

### Language

| Value | Records | Share |
|---|---:|---:|
| en | 3600 | 97.72% |
| und | 84 | 2.28% |

### Domain

| Value | Records | Share |
|---|---:|---:|
| Alphanumeric Substitution | 5 | 0.14% |
| Anagramming | 5 | 0.14% |
| Base Encoding | 5 | 0.14% |
| Blocking Internet Connection | 5 | 0.14% |
| Bringing Down Hosts and Servers (Denial of Service) | 5 | 0.14% |
| Business Intelligence | 5 | 0.14% |
| Clickbait | 5 | 0.14% |
| Common Queries | 126 | 3.42% |
| Compromising Computers | 5 | 0.14% |
| Content Creation | 5 | 0.14% |
| Conversational Agent | 5 | 0.14% |
| Cookie Theft | 5 | 0.14% |
| Corrupting an Operating System | 5 | 0.14% |
| Crippling Critical Infrastructures | 5 | 0.14% |
| Cryptocurrency Mining | 5 | 0.14% |
| Data Eavesdropping | 5 | 0.14% |
| Data Security Harm | 374 | 10.15% |
| Device and Driver Enumeration | 5 | 0.14% |
| Dumpster Diving | 5 | 0.14% |
| Emoji Substitution | 5 | 0.14% |
| Encrypting Documents and Demanding Ransom (Ransomware) | 5 | 0.14% |
| Entertainment | 5 | 0.14% |
| Environment Variable Analysis | 5 | 0.14% |
| Exploiting System Vulnerabilities | 5 | 0.14% |
| Financial Data | 204 | 5.54% |
| Financial Harm | 306 | 8.31% |
| Homophonic Substitution | 5 | 0.14% |
| Information Dissemination | 5 | 0.14% |
| Information Retrieval | 5 | 0.14% |
| Instruction | 5 | 0.14% |
| Introduce System Fingerprinting | 5 | 0.14% |
| Keylogging | 5 | 0.14% |
| Language Translation | 10 | 0.27% |
| Learning and Tutoring | 5 | 0.14% |
| Malware Distribution | 5 | 0.14% |
| Marketing & Advertising | 5 | 0.14% |
| Memory Scanning | 5 | 0.14% |
| Misinformation & Propaganda | 5 | 0.14% |
| Misspelling Intentionally | 5 | 0.14% |
| Multilingual | 84 | 2.28% |
| Network Propagation | 5 | 0.14% |
| Others | 510 | 13.84% |
| Persuasion | 5 | 0.14% |
| Physical Data | 374 | 10.15% |
| Physical Harm | 340 | 9.23% |
| Programming Help | 5 | 0.14% |
| Research Assistance | 5 | 0.14% |
| Reverse Text | 5 | 0.14% |
| Scams & Fraud | 5 | 0.14% |
| Screen Scraping | 5 | 0.14% |
| Sending Out Spam Emails | 5 | 0.14% |
| Sentiment Analysis | 5 | 0.14% |
| Social Interaction | 5 | 0.14% |
| Space Removal & Grouping | 5 | 0.14% |
| Substitution Ciphers | 5 | 0.14% |
| Task Automation | 5 | 0.14% |
| Technique Queries | 87 | 2.36% |
| Traffic Analysis | 5 | 0.14% |
| Virtual Creation | 42 | 1.14% |
| banking | 144 | 3.91% |
| duplicate_sentence_detection | 4 | 0.11% |
| goal_hikacking_attacks | 5 | 0.14% |
| grammar_correction | 4 | 0.11% |
| hate_detection | 4 | 0.11% |
| natural_language_inference | 4 | 0.11% |
| prompt_leaking_attacks | 5 | 0.14% |
| sentiment_analysis | 4 | 0.11% |
| slack | 105 | 2.85% |
| spam_detection | 4 | 0.11% |
| summarization | 4 | 0.11% |
| travel | 140 | 3.80% |
| workspace | 560 | 15.20% |

### Attack family

| Value | Records | Share |
|---|---:|---:|
| NONE | 339 | 9.20% |
| PROMPT_INJECTION | 3345 | 90.80% |

### Attack objective

| Value | Records | Share |
|---|---:|---:|
| NONE | 1288 | 34.96% |
| PROMPT_OR_POLICY_DISCLOSURE | 5 | 0.14% |
| SENSITIVE_DATA_EXFILTRATION | 1088 | 29.53% |
| TASK_HIJACKING | 283 | 7.68% |
| UNAUTHORIZED_ACTION | 2108 | 57.22% |

### Delivery mode

| Value | Records | Share |
|---|---:|---:|
| DIRECT | 349 | 9.47% |
| INDIRECT | 3335 | 90.53% |

### Annotation tier

| Value | Records | Share |
|---|---:|---:|
| DETERMINISTIC_MAPPING | 10 | 0.27% |
| HEURISTIC_MAPPING | 3335 | 90.53% |
| UNANNOTATED | 339 | 9.20% |

### Benchmark suite

| Value | Records | Share |
|---|---:|---:|
| banking | 144 | 3.91% |
| slack | 105 | 2.85% |
| travel | 140 | 3.80% |
| workspace | 560 | 15.20% |

### User tool

| Value | Records | Share |
|---|---:|---:|
| AmazonGetProductDetails | 124 | 3.37% |
| EvernoteManagerSearchNotes | 124 | 3.37% |
| GitHubGetRepositoryDetails | 124 | 3.37% |
| GitHubGetUserDetails | 124 | 3.37% |
| GitHubSearchRepositories | 124 | 3.37% |
| GmailReadEmail | 124 | 3.37% |
| GmailSearchEmails | 124 | 3.37% |
| GoogleCalendarGetEventsFromSharedCalendar | 124 | 3.37% |
| GoogleCalendarReadEvents | 124 | 3.37% |
| ShopifyGetProductDetails | 124 | 3.37% |
| TeladocViewReviews | 124 | 3.37% |
| TodoistSearchTasks | 124 | 3.37% |
| TwilioGetReceivedSmsMessages | 124 | 3.37% |
| TwitterManagerGetUserProfile | 124 | 3.37% |
| TwitterManagerReadTweet | 124 | 3.37% |
| TwitterManagerSearchTweets | 124 | 3.37% |
| WebBrowserNavigateTo | 124 | 3.37% |

### Attacker tool

| Value | Records | Share |
|---|---:|---:|
| AmazonViewSavedAddresses | 34 | 0.92% |
| AmazonViewSavedPaymentMethods | 34 | 0.92% |
| AugustSmartLockGrantGuestAccess | 34 | 0.92% |
| AugustSmartLockUnlockDoor | 34 | 0.92% |
| AugustSmartLockViewAccessHistory | 34 | 0.92% |
| BankManagerGetAccountInformation | 34 | 0.92% |
| BankManagerPayBill | 34 | 0.92% |
| BankManagerSearchPayee | 34 | 0.92% |
| BankManagerTransferFunds | 34 | 0.92% |
| BinanceDeposit | 34 | 0.92% |
| BinanceGetOrderHistory | 34 | 0.92% |
| BinancePlaceOrder | 34 | 0.92% |
| BinanceWithdraw | 34 | 0.92% |
| CiscoUmbrellaCreatePolicy | 34 | 0.92% |
| CiscoUmbrellaGetLogDetails | 34 | 0.92% |
| CiscoUmbrellaUpdatePolicy | 34 | 0.92% |
| DeepfakeGeneratorGenerateAudioDeepfake | 34 | 0.92% |
| DropboxGetItemMetadata | 34 | 0.92% |
| DropboxListFilesAndFolders | 34 | 0.92% |
| DropboxMoveItem | 34 | 0.92% |
| EmergencyDispatchSystemCreateDispatchRequest | 34 | 0.92% |
| EpicFHIRDownloadFiles | 34 | 0.92% |
| EpicFHIRGetPatientDetails | 34 | 0.92% |
| EpicFHIRManageAppointments | 34 | 0.92% |
| EpicFHIRManageClinicalDocuments | 34 | 0.92% |
| EpicFHIRManagePatientRecords | 34 | 0.92% |
| EvernoteManagerManageNoteSharingLevel | 34 | 0.92% |
| ExpediaSearchReservations | 34 | 0.92% |
| FacebookManagerGetUserProfile | 34 | 0.92% |
| FacebookManagerSearchFriends | 34 | 0.92% |
| FacebookManagerUpdateUserProfile | 34 | 0.92% |
| FedExShipManagerManageShipment | 34 | 0.92% |
| FedExShipManagerSearchShipment | 34 | 0.92% |
| GitHubDeleteRepository | 34 | 0.92% |
| GitHubGetUserDetails | 34 | 0.92% |
| GmailSendEmail | 1088 | 29.53% |
| GoogleHomeScheduleDeviceAction | 34 | 0.92% |
| GoogleMapGetCurrentLocation | 34 | 0.92% |
| GoogleSearchGetSearchHistory | 34 | 0.92% |
| IndoorRobotGoToRoom | 34 | 0.92% |
| InvestmentManagerGetHoldingsInformation | 34 | 0.92% |
| NortonIdentitySafeAutoFillPersonalInfo | 34 | 0.92% |
| NortonIdentitySafeDisableTwoFactorAuthentication | 34 | 0.92% |
| NortonIdentitySafeSearchPasswords | 34 | 0.92% |
| NortonIdentitySafeSharePassword | 34 | 0.92% |
| SlackLeaveChannel | 34 | 0.92% |
| SpokeoDownloadPublicRecord | 34 | 0.92% |
| SpokeoSearchPeople | 34 | 0.92% |
| TDAmeritradeGetHoldingsInformation | 34 | 0.92% |
| TeladocAccessUserInfo | 34 | 0.92% |
| TeladocViewPrescriptions | 34 | 0.92% |
| TerminalExecute | 34 | 0.92% |
| The23andMeDownloadData | 34 | 0.92% |
| The23andMeGetGeneticData | 34 | 0.92% |
| The23andMeGetHealthRecommendations | 34 | 0.92% |
| The23andMeShareData | 34 | 0.92% |
| The23andMeUpdateData | 34 | 0.92% |
| TrafficControlGivePriorityToEmergencyVehicle | 34 | 0.92% |
| TrafficControlManageTrafficLightState | 34 | 0.92% |
| TwilioGetPhoneNumberInfo | 34 | 0.92% |
| VenmoListLinkedAccounts | 34 | 0.92% |
| VenmoWithdrawMoney | 34 | 0.92% |
| WebBrowserSearchHistory | 34 | 0.92% |

### InjecAgent attack category

| Value | Records | Share |
|---|---:|---:|
| Data Security Harm | 374 | 10.15% |
| Financial Data | 204 | 5.54% |
| Financial Harm | 306 | 8.31% |
| Others | 510 | 13.84% |
| Physical Data | 374 | 10.15% |
| Physical Harm | 340 | 9.23% |

### InjecAgent setting

| Value | Records | Share |
|---|---:|---:|
| base | 1054 | 28.61% |
| enhanced | 1054 | 28.61% |

### AgentDojo suite

| Value | Records | Share |
|---|---:|---:|
| banking | 144 | 3.91% |
| slack | 105 | 2.85% |
| travel | 140 | 3.80% |
| workspace | 560 | 15.20% |

### Mapping confidence

| Value | Records | Share |
|---|---:|---:|
| 0.00 | 339 | 9.20% |
| 0.55 | 949 | 25.76% |
| 0.70 | 250 | 6.79% |
| 0.80 | 2136 | 57.98% |
| 1.00 | 10 | 0.27% |

## Mapping quality

| Indicator | Records | Rate |
|---|---:|---:|
| Gold source | 0 | 0.00% |
| Deterministic mapping | 10 | 0.27% |
| Heuristic mapping | 3335 | 90.53% |
| Unannotated | 339 | 9.20% |
| Requires manual review | 3674 | 99.73% |
| Review candidates | 3684 | 100.00% |
| Review queue after threshold | 3684 | 100.00% |
| Missing mapping confidence | 0 | 0.00% |

## UNKNOWN and UNDETERMINED fields

| Field | UNKNOWN | UNDETERMINED | Combined | Rate |
|---|---:|---:|---:|---:|
| annotations.authority_status | 1288 | 0 | 1288 | 34.96% |
| annotations.instruction_addressee | 339 | 0 | 339 | 9.20% |
| annotations.instruction_presence | 0 | 339 | 339 | 9.20% |
| annotations.instruction_presentation | 339 | 0 | 339 | 9.20% |
| annotations.protected_policy_alignment | 0 | 3684 | 3684 | 100.00% |
| annotations.user_goal_alignment | 0 | 1288 | 1288 | 34.96% |
| content.content_origin | 178 | 0 | 178 | 4.83% |
| content.delivery_mode | 0 | 0 | 0 | 0.00% |
| content.ingestion_path | 0 | 0 | 0 | 0.00% |
| content.source_integrity | 1576 | 0 | 1576 | 42.78% |
| content.source_role | 0 | 0 | 0 | 0.00% |

## Lengths

| Measure | Count | Min | Mean | Median | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Content characters | 3684 | 11 | 173.600163 | 162.0 | 366 | 593 |
| Content UTF-8 bytes | 3684 | 17 | 174.726384 | 162.0 | 366 | 593 |
| Spans per record | 3684 | 0 | 1.480185 | 2.0 | 2 | 2 |
| Span length in characters | 5453 | 17 | 160.904456 | 148.0 | 306 | 593 |

## Span types

| Value | Records | Share |
|---|---:|---:|
| DIRECTIVE | 3057 | 56.06% |
| INJECTION_PAYLOAD | 2396 | 43.94% |
