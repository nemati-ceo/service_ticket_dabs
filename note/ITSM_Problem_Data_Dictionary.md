# ITSM Problem - Data Dictionary with Release Notes

**Data Domain:** ServiceNow Problem Table
**Object Type:** Dataflow
**Location:** ODA_Sponsored_ITSM_Prod
**Data Domain:** Service Management Data (ITSM)

### DEFINITION
ITSM Problem management is a process of identifying and managing the cause of incidents on IT services. Its goal is to minimize impact on business processes and to restore IT services by identifying and analyzing the cause of recurring issues then finding permanent solutions.

### INTENT
The data provided in this dataset has been prepared to build reports around trending, key performance indicators, and as a means to enrich other datasets that may expose a common root cause or technical debt.
This data is not intended to be an alternative to real-time data provided directly in ServiceNow. If you need real-time data, please reach out to #servicenow-reporting for better options.

### SOURCE OF TRUTH
The ServiceNow Problem module is the source of truth for this dataset.

### SCHEDULED REFRESHES
This data refreshes daily to show the previous day's results. Historical data for Problem records is as early as 03/08/2021.

---

### DATA DICTIONARY

| Dataset Field Name | Description | Underlying ServiceNow View Name | Data Type | Reporting Guidance |
| :--- | :--- | :--- | :--- | :--- |
| `prb_assignment_group` | The ServiceNow team responsible for handling specific tasks. | Assignment group | string | A Problem can be assigned to more than one group during its lifecycle, however the Assignment Group associated with the problem will be the final one captured and recorded on the record. |
| `prb_assignment_group_link` | | | | 07/01/2025 Currently contains no data. |
| `prb_assignment_group_sys_id` | This is a 32-character, alphanumeric identifier associated with the ServiceNow assignment group. | Assignment group system ID | string | |
| `prb_assignment_group_value` | | | | 07/01/2025 Currently contains no data. |
| `prb_bug_category` | Type of bug associated with the Jira issue. | Category | string | The category type is predetermined by Jira. |
| `prb_bus_app` | Business Application associated with the Problem | Business application | string | 07/01/2025 Currently contains no data. |
| `prb_business_service` | Identifies the NM business service experiencing issues. For example, Email/Calendaring, Batch Operations, Network Services. | Service | string | |
| `prb_cause` | Root cause category for the problem. | Cause | string | |
| `prb_cause_notes` | Free form updates to the problem record to document root cause | Cause Notes | string | |
| `prb_change_request` | ServiceNow change records that are associated to the specific ServiceNow problem record. | Change Request | string | |
| `prb_ci` | Drop down list of configuration items (CI) affected by the issues the problem is addressing. | Configuration item | string | |
| `prb_close_notes` | Free form field for updates when closing the problem record. | Close notes | string | |
| `prb_close_date` | Close Date for the problem. | Close Date | datetime | Tracks when the record is closed, not necessarily when teams believe the issue is resolved. Close date will populate 60 days after a resolved date has been populated. |
| `prb_created_by` | The NM full name of the person who created the problem record. | Created by | string | |
| `prb_created_by_nmid` | The NM ID associated to the person who created the problem record. | Created by ID | string | |
| `prb_created_date` | The date and time a problem record itself is created in the table. | Created | datetime | This date/time stamp does not change. |
| `prb_desc` | Description of the issue | Description | string | |
| `prb_director_email` | The NM email address for the Director assigned to the problem. | Director email | string | |
| `prb_director_name` | The team director whose team is working to resolve the problem | Director | string | |
| `prb_jira_assignee` | The NM Full Name of the person who is assigned to the Jira issue. | Jira Assignee | string | |
| `prb_jira_issue_id` | The unique alphanumeric indicator associated with an issue. | Jira Issue ID | string | |
| `prb_jira_project_id` | The name of the Jira initiative the work is associated to. | Jira Project | string | |
| `prb_jira_state` | Indicates the status within the lifecycle of a Jira issue. | Jira State | string | |
| `prb_jira_ticket` | Jira Ticket stories tracking work related to the problem | Jira ticket | string | |
| `prb_jira_type` | This indicator will either be blank or contain "Problem". | Jira issue type | string | |
| `prb_jira_url` | Jira URL linking from the problem record to Jira | Jira URL | string | |
| `prb_known_error` | Known Error is used when a problem owner implements a work around and creates a knowledge article. | Known Error | string | |
| `prb_major_problem` | True/False indicator that identifies if the problem record is designated as Major | Major Problem | string | |
| `prb_manager_email` | NM email address for the Manager assigned to the problem. | Manager email | string | |
| `prb_manager_name` | The team manager whose team is working to resolve the problem | Manager | string | |
| `prb_number` | The unique "ticket" number assigned to a record that can be used to identify, track, store and retrieve a record over time. | Number | string | |
| `prb_primary_known_error_article` | The Knowledge Base article used to resolve the problem. | Primary Known Error Article | string | |
| `prb_primary_known_error_article_number` | The Knowledge Base article number. | Primary Known Error Number | string | |
| `prb_priority` | Priority considers the urgency of the issue and the impact (how many are affected) | Priority | int64 | |
| `prb_proactive_problem` | True/False indicator that shows if a problem was opened without preceding incidents occurring. | Proactive Problem | string | In ServiceNow, a proactive problem is on it's own record resulting in a reported incident. |
| `prb_rca_date` | | RCA date | datetime | |
| `prb_related_incidents` | The distinct count of incidents associated with the problem. | Related Incidents | string | |
| `prb_resolution_code` | Resolved disposition for the problem. | Resolution code | string | |
| `prb_resolved_date` | Resolved date and time indicates when the issue is fixed | Resolved | datetime | In ServiceNow, the prb_resolved_date field behaves weird... problem was closed as a duplicate, was cancelled... problems was given of a status of "Risk Accepted" |
| `prb_service_feature` | The specific feature of a service being affected. For example, Policy Change within Servicing, enforce within Illustrations | Service feature | string | |
| `prb_short_desc` | Short Description that captures the problem statement | Problem statement | string | |
| `prb_state` | Indicates the status within the lifecycle of a problem, ex. Open, Resolved, Closed. | Problem state | string | |
| `prb_sub_cause` | Additional causal information based on the problem Cause Code. For example, outage, unintended action, missing requirements, etc. | Sub Cause Code | string | |
| `prb_sys_id` | The system ID for the RCA associated with the Problem record. | Sys ID | string | |
| `prb_tags` | A label used to organize and categorize records. | Tags | string | |
| `prb_updated_date` | Tracked date and time when a problem record is edited. | Updated | datetime | |
| `prb_user_input` | String containing user input | User Input | string | |
| `prb_workaround_applied` | True/False indicator showing whether a workaround was applied in order to close the problem record. | Workaround Applied | string | |
| `rca_business_duration_sec` | Business duration, in seconds, from when a Problem record is opened until it has been diagnosed with a root cause. | RCA business duration | int64 | |
| `rca_sla_percentage` | | | | |
| `rca_duration_sec` | Duration, in seconds, from when a Problem record is opened until it has been diagnosed with a root cause. | | int64 | |
| `rca_sla_breached` | | | | |
| `rca_sla_name` | | | | |
| `rca_sla_percentage` | The percentage of time the Problem record accrued against the Root Cause Analysis (RCA) SLA. | RCA SLA | double | If a Problem record met its SLA, the percentage will be under 100%. If it failed the SLA, the percentage will be over 100%. |
| `res_business_duration_sec` | | | | |
| `res_business_percentage` | The percentage of business time the Problem record accrued against the Resolution SLA. | | double | If a Problem record met its SLA, the percentage will be under 100%. If it failed the SLA, the percentage will be over 100%. |
| `res_duration_sec` | | | | |
| `res_sla_breached` | True/False result indicating if the Problem was resolved within the SLA timeframe. | | string | This SLA timer starts when the Problem is opened... therefore will show SLA breached status if it exceeds its business day limit. Once the SLA breached, it remains regardless of the Problem 'state' (status)... SLA Breached status to "True". When changed to "Resolved" the timer stops... 'state' changes to "Closed" the time locked.<br>Priority 1, 2, 3 Problems have defined SLAs... priority 4, 5 Problems do not have SLAs... minimal Priority 4 and 5 problems if they were initially 2 or 3 then changed. |
| `res_sla_name` | Resolution SLA definition by priority. | | string | Applies to Priority 1, 2, 3 Problems. Priority 4 and 5 do not have SLAs. |
