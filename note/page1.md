# Service Management Data (ITSM)

Created by Squier, Renee, last updated on Sep 03, 2025 • 3 minute read

* Intended Use of this Data
* Who is providing this data?
* Notes About ServiceNow Data
* Getting Started
* Step 1 – Evaluate your requirements
* Step 2 – Get Access
* Step 3 – Start Building


* Data Dictionaries with Release Notes
* What if I don't see what I need?
* Contact Us

## Intended Use of this Data

This sponsored data provides historical IT Service Management (ITSM) data. Its intent is to provide data for BI Reporters interested in KPI trending or as enrichment information that, for example, identifies a common root cause or exposes technical debt.

## Who is providing this data?

This dataflow in compliance is being provided in compliance with the Sponsored Dataset Contract by the ODA team in partnership with the ServiceNow platform team.

## Notes About ServiceNow Data

This sponsored data contains provides the daily breakdowns of Incident, Problem, and Change results at the record level.

**REFRESH RATES**
In addition to the historical data, new data is continually added and is refreshed daily to show the previous day's results.

* Step 1 - At 3:00AM, the prior day's data for any activity between 12:00:00AM - 11:59:59PM is extracted via an api connection and stored in our data lake house.
* Step 2 - At 5:00 AM, this sponsored data within the lake house is refreshed.
* Step 3 - At 6:00 AM, the data flows within the Power BI services are refreshed.

## Getting Started

We've laid out a few items to review before diving in:

### Step 1 – Evaluate your requirements

**For example, do you need any of the following:**

* [ ] Incident
* [ ] Assignment Group
* [ ] Priority
* [ ] Respond / Resolve SLA Met
* [ ] Time To Information
* [ ] Configuration Item
* [ ] Service


* [ ] Problem
* [ ] Owned by Team
* [ ] Priority
* [ ] Status
* [ ] Expected Close date met
* [ ] Root Cause Analysis
* [ ] Cause
* [ ] Configuration Item


* [ ] Change
* [ ] Owned by Team
* [ ] Type
* [ ] Risk
* [ ] Priority
* [ ] Date and Times
* [ ] Final Disposition



### Step 2 – Get Access

Once you have evaluated your requirements and determined the Service Management datasets will meet your needs, we can start the steps to get you access to our data:

* Please read the Dataset Contracts to understand what we support and what the expectations are for consumers of this data.
* To gain access to Service Management data, please complete an iRequest to be added to the following Active Directory group:
* WG-ODA-SPData-ITSM



### Step 3 – Start Building

As a member of the AD groups in step 2, you can now access the workflow as follows

* Workspace: **ODA_Sponsored_ITSM_Prod**

**Workflows**

* ocio_dw_servicenow_assgngrp_data_v1
* ocio_dw_bus-app-spt_data_v1
* ocio_dw_servicenow_change_data_v1
* ocio_dw_servicenow_incident_data_v1
* ocio_dw_servicenow_incident_requests_data_v1
* ocio_dw_servicenow_ITAM_data_v1
* ocio_dw_servicenow_problem_data_v2
* ocio_dw_servicenow_req_items_data_v1
* ocio_dw_servicenow_task_data_v1
* ocio_dw_servicenow_workstation_data_v1
* oda_knowledge_base_v1
* oda_servicenow_interactions_v1
* Select and include entities you need to enrich your model
* Since each row is uniquely identified, there can be a one-to-one or one-to-many relationship to your dataset
* Roll up your data or build data slicers using dimensional fields

## Data Dictionaries with Release Notes

All of our ITSM sponsored data is provided as dataflows within the **ODA_Sponsored_ITSM_Prod** workspace in Power BI. If your reporting needs require accessing the data outside of the Power BI solution, we can help. Please reach out to us on Slack, at # oda-sponsored-datasets or complete our ODA Intake Request Form.

Select the following links below to route to release information and data dictionaries.

* ITSM Assignment Groups - Data Dictionary with Release Notes
* ITSM Change - Data Dictionary with Release Dates
* ITSM Incident - Data Dictionary with Release Notes
* ITSM Incident Type = Requests - Data Dictionary with Release Notes
* ITSM Interactions - Data Dictionary with Release Dates
* ITSM Knowledge - Release Notes and Data Dictionary
* ITSM Problem - Data Dictionary with Release Notes
* Business Applications Service - Data Dictionary and Release Notes
* ITSM ITAM - Data Dictionary with Release Dates (Servers)
* CMDB Workstations - Data Dictionary with Release Dates
* ITSM Tasks - Data Dictionary with Release Dates

## What if I don't see what I need?

Our goal is to provide data attributes that can be used by multiple reporting teams for multiple reasons.

If you need:

* Retrospective, convenient, consumable, and trusted versions of data for BI reporting and analytical needs
* Need Service Management data along with other data sources for your reporting
* Use our Sponsored Data and/or ServiceNow, but are not finding what you need,

Then, contact the OCIO Data & Analytics team, at this preferred Slack channel, **# oda-sponsored-datasets**.

If you need:

* Real-time access to dashboards to understand the status of an issue, change, task, etc.
* If you access ServiceNow directly and need help finding a specific dashboard or report,

Then contact the Service Management and Shared Operations team, at this preferred Slack channel, **#servicenow-reporting**.

## Contact Us

Have questions or comments about this dataset? Join the Sponsored Dataset Slack Channel: #oda-sponsored-datasets