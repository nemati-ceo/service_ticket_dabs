# ITSM Incident - Data Dictionary with Release Notes

**Created by:** Squier, Renee, last updated on Mar 20, 2026  · 8 minute read

**Data Domain:** ServiceNow Incident Table  
**Object Type:** Dataflow  
**Location:** ODA_Sponsored_ITSM_Prod  
**Data Domain:** Service Management Data (ITSM)

**CONTACT US** **Have questions or comments?** Please reach out to our `#oda-sponsored-datasets` Slack channel.

---

## DEFINITION
In Service Management, incidents are unplanned interruptions or reductions in the quality of services. It includes service failures and degradation that affects its delivery to Users. 
In ITIL the incident management process includes logging the issue, categorizing, prioritizing, and resolving issues.

**NOTE:** Incidents and the management of them is different than Service Requests which are user-initiated requests for information or access to services. If you are interested in Service Requests, please see **Incident Type = Request** and **ITSM Service Catalog Requests**.

## INTENT
The data provided in this dataset has been prepared to build reports around trending, key performance indicators, and as a means to enrich other datasets that may expose a common root cause or technical debt. 
This data is not intended to be an alternative to real-time data provided directly in ServiceNow. If you need real-time data, please reach out to `#servicenow-reporting` for better options.

## SOURCE OF TRUTH
The ServiceNow Incident module is the source of truth for this dataset.

## SCHEDULED REFRESHES
This data refreshes daily to show the previous day's results. Comprehensive historical data for Incidents is as early as 01/01/2023.

---

## DATA DICTIONARY

| Dataset Field Name | Description | Underlying ServiceNow View Name | Data Type | Reporting Guidance |
| :--- | :--- | :--- | :--- | :--- |
| `call_script_used` | True/False attribute that indicates if a call script was used when handling the incident. | Tag | string | The "call_script_used" is taken from the incident Tag fi... sponsored dataflow contains its own column that indica... when was tagged to the record, False when it was not. |
| `chg_req_number` | Associated Change record number initiated as a result of the incident's root cause. | Change Request | string | |
| `etl_load_datetime` | This is data provided as part of this team's extract, transform, load tasks. It lets the BI reporter know data quality timeliness. | NA | datetime | This is data provided as part of this team's extract, tran... load tasks and is not found in the source of truth. |
| `incident_analyzed` | True/False indicator that marks an incident as being reviewed as part of the Problem Management process. | Tag | string | The "Incident_analyzed" is taken from the Incident Tag... The sponsored dataflow contains its own column that i... True when was tagged to the record, False when it was... This is a specific tag used by TCS for Problem Manage... tracking. |
| `inc_assigned_to_email` | Northwestern Mutual email address for the person working to resolve the incident. | Assigned to email | string | The sponsored data provides the email of the last pers... working to resolve the incident. |
| `inc_assigned_to_name` | The person working to resolve the incident at any given time. An incident can be assigned to more than one person throughout its lifecycle; however, the person who resolves the incident will be the one reported. | Assigned to | string | |
| `inc_assignment_group` | The team working to resolve the incident at any given time. An incident can be assigned to more than one team throughout its lifecycle; however the team that resolves the incident will be the one reported. | Assignment group | string | The sponsored data provide the last team who owned... incident. |
| `inc_assignment_group_sys_id` | This is a 32-character, alphanumeric identifier associated with the ServiceNow assignment group, which is the team that supports a service or asset. | Assignment Group System ID | string | |
| `inc_business_service` | Business service affected by the incident. | Service | string | |
| `inc_caller_id_email` | Email of the individual affected by the issue. | Email | string | |
| `inc_caller_id_name` | Individual who is affected by the issue. | Caller | string | |
| `inc_caused_by` | Cause of the Incident | Cause | string | |
| `inc_ci_name` | The Configuration Item is the asset affected by the issue. | Configuration Item | string | |
| `inc_ci_utan` | The UTAN associated with the Configuration Item | UTAN | string | |
| `inc_close_notes` | Resolution notes describing what was done to resolve the incident. | Resolution notes | string | The field used to document steps taken to resolve the i... |
| `inc_contact_type` | The media used to open an incident. For example, via an alert, chat, email, phone, walk-up | Contact Type | string | |
| `inc_cost_center_code` | Four-digit cost center of the person that reported the issue | Caller Cost Center Code | string | |
| `inc_created_by_team` | The team that receives the initial contact for the incident. | Created by team | string | |
| `inc_escalation` | Escalations indicates if an incident was expedited | Escalation | string | The Sponsored data will show a result of: Normal, Moderate, High, or Overdue |
| `inc_event_key` | The Event Key identifies which process created an auto-ticket. | Event key | string | |
| `inc_fcr` | First Contact Resolution (FCR). Boolean indicator of Yes/No to marking if the incident was resolved by the first person assigned. | FCR | string | The Sponsored data will show a result of: NA, No, or Y... NA signifies that the ticket does not have the ability to resolved by the first contact so should not be used in F calculations and it would negatively impact. For more, Contact Resolution (FCR) - Technology Customer Succ Engineering |
| `inc_initial_service` | The initial service identified when the incident is created. | Initial Service | string | The initial service may be used by reporters to underst... misdiagnosed symptoms and/or why an Assignment Gr... originally assigned, then transfers it. |
| `inc_initial_service_feature` | The initial service feature that the created by team identified as being affected. | Initial Service Feature | string | |
| `inc_initial_type` | The first incident type used when the ticket was opened. | Initial Type | string | This field is used to understand the initial type used wh... creating an incident. Reporters may use this field to un... |
| `inc_major_state` | Major incidents are those that have the potential to become P1 incidents if they are not immediately addressed. Major incidents are managed by the Critical Incident Response Team. | Major incident state | string | The major incident field will have an entry of 'accepted' 'rejected' or a blank entry if false. |
| `inc_major_time_to_engage` | Duration in seconds that it took to reach the Engage status of a major incident. | Time to Engage | int64 | This result has been translated into the time fundamen... dimension unit of seconds. |
| `inc_major_time_to_mitigate` | Duration in seconds that it took to reach the Mitigate status of a major incident. | Time to Mitigate | int64 | This result has been translated into the time fundamen... dimension unit of seconds. |
| `inc_number` | The unique "ticket" number assigned to a record that can be used to identify, track, store and retrieve a record over time | Number | string | |
| `inc_opened_at` | The date and time an incident record was created. This is the time stamp used to measure the start of a response, mitigate and/or resolve, and close times. | Opened | datetime | This dataset contains comprehensive, historical data a... 01/01/2023. ServiceNow as the source of truth contain... comprehensive historical data as of 03/06/2021. |
| `inc_opened_by_email` | The email of the person that created the incident record. | Opened By Email | string | |
| `inc_opened_by_name` | The person that created the incident record. | Opened By | string | |
| `inc_parent_incident` | Parent incidents are used to group incidents of the same nature together. When the Parent incident is resolved, unresolved Child incidents inherit the resolution. | Parent incident | string | When the Parent incident is resolved, unresolved Child incidents inherit the resolution. Each incident follows a auto-close. |
| `inc_priority` | Priority indicates the timescales and effort to respond and resolve (or mitigate) an issue. It is determined using an impact and urgency matrix. | Priority | int64 | |
| `inc_reopen_count` | Number of times an incident is reassigned after it has been put into a 'Resolved' state. | Reopen | int64 | An incident may be reopened if it is in the 'Resolved' st... |
| `inc_resolution_code` | Resolution codes are used to indicate the status of the incident that pauses SLA clock and can be confirmed by the end user | Resolution code | string | |
| `inc_resolved_at` | Date and Time Incident is Resolved | Time of Resolve | datetime | |
| `inc_service_feature` | The service feature that was affected | Service feature | string | |
| `inc_short_description` | Free form explanation describing the issue. | Short Description | string | |
| `inc_stage` | Indicator if the incident is: completed, in progress, or paused. | Stage | string | |
| `inc_state` | Indicates the status within the lifecycle of an incident, ex. Open, Resolved, Closed. | State | string | |
| `inc_type` | The category for the issue being addressed. | Incident Type | string | |
| `inc_u_knowledge_used` | Tag used by Agents to indicate when a knowledge document was used to resolve an incident. | Tag | string | |
| `inc_u_time_of_detection` | The time stamp in which the incident is observed. | Time of Detection | datetime | |
| `inc_u_time_of_engage` | The time stamp in which the incident remediation process starts. | Time of Engage | datetime | |
| `inc_u_time_of_impact` | The time stamp in which the incident affects the IT environment | Time of Impact | datetime | |
| `inc_u_time_of_know` | The time stamp in which the incident is observed. | Time of Know | datetime | |
| `inc_u_time_of_mitigate` | The time stamp certain incidents are considered to be in a resolved or fixed state and the threat or impact is contained. | Time of Mitigate | datetime | |
| `inc_u_time_of_start` | The time stamp referring to the exact, recorded time the incident process commences. | Time of Start | datetime | |
| `inc_u_time_to_detect` | The measurement of the time from the onset of an incident until it is discovered. | Time to Detect (TDD) | int64 | |
| `inc_u_time_to_resolve` | The measurement of time from the time of engagement through to the time of resolution. | Time to Resolve (TTR) | int64 | |
| `inc_updated_by` | Updated By field captures the LAN ID for the person, or the system, that last updated the ticket | Updated By | string | |
| `inc_zendesk_knowledge_id` | The Zendesk knowledge identification number used to troubleshoot or resolve the incident. | Zendesk knowledge ID | string | |
| `inc_zendesk_ticket_number` | The corresponding Zendesk ticket associated with the ServiceNow Incident. | Zendesk ticket number | string | |
| `Jira_key` | The Jira Issue, or story, that is related to the Incident record in ServiceNow. | Jira issue ID | string | In ServiceNow, what ODA has labeled the Jira-key is ca... 'Jira issue ID'. ODA has delineated this as the key beca... there is a field within Jira source data that is named 'Is... which represents a different data attribute. |
| `prb_number` | This field references a specific Problem record number associated with the incident | Problem | string | |
| `resolution_business_duration` | A timestamp that displays the duration, in seconds, that an incident has been opened using business hours. Its purpose is to time the incident against respond and resolve SLAs. | Business elapsed time | int64 | This result has been translated into the time fundamen... dimension unit of seconds. |
| `resolution_duration` | A timestamp that displays the duration, in seconds, that an incident has been opened. Its purpose is to time the incident against respond and resolve SLAs. | Elapsed time | int64 | This result has been translated into the time fundamen... dimension unit of seconds. |
| `resolution_sla_breached` | TRUE or FALSE result of a respond and resolve SLA. | Resolution SLA breached | string | |
| `resolution_sla_business_percentage` | The percentage of business time the Incident against the Resolution SLA. | SLA business percentage | double | |
| `resolution_sla_name` | The expected SLA resolution time based on the priority of an incident. | SLA name | string | This data has been formatted from row format to colum... avoid duplicate incident counts. |
| `response_business_duration` | A timestamp that displays the business duration, in seconds, that an incident has been opened. Its purpose is to time the incident against respond SLAs. | Business elapsed time | int64 | This result has been translated into the time fundamen... dimension unit of seconds. |
| `response_duration` | A timestamp that displays the duration, in seconds, that an incident has been opened. Its purpose is to time the incident against respond and resolve SLAs. | Actual elapsed time | int64 | This result has been translated into the time fundamen... dimension unit of seconds. |
| `response_sla_breached` | TRUE or FALSE result of a respond and resolve SLA | Has breached | string | |
| `response_sla_name` | The expected SLA response time based on the priority of an incident. | SLA definition | string | The ODA Data Engineering team has reformatted the ServiceNow Incident SLA data from row format to colu... show SLA results based on the priority of the incident w... was closed. This has been done to avoid duplicate inci... counts. |
| `response_sla_business_percentage` | (business duration/target) | SLA business percentage | double | |
| `response_sla_percentage` | Response duration/target | SLA percentage | double | |
| `task_reassignment_count` | A count representing then number of times an Incident was reassigned to a different Assignment Group. | Reassignment count | string | |
