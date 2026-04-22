Project Title
A Predictive Analytics System for E-commerce Growth and Inventory Management
Project Type
MSc Group Project (SaaS System + ML + Full Stack)
Objective
Build a production-style SaaS platform that:
Processes e-commerce transaction data via an ETL pipeline
Generates predictive insights (sales forecasting & inventory risk)
Serves predictions via backend APIs
Displays insights in a real-time dashboard
Evaluates both accuracy (MAE, RMSE) and system latency (<200ms target)
2. CORE SYSTEM GOAL (VERY IMPORTANT)
This is NOT just a machine learning project.
We are building:
A complete end-to-end predictive SaaS system that integrates
ETL → ML → Backend API → Frontend Dashboard → Evaluation
3. DATASET
We will use:
UCI Online Retail Dataset
Data Characteristics
Transaction-level data
Fields: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
Important Rules
Negative quantities = returns
InvoiceNo starting with "C" = cancellation
Revenue = Quantity × UnitPrice
Data must be aggregated before modeling
4. FINAL OUTPUT OF SYSTEM
The system must provide:
A. Forecasting
Predict future demand (daily or 7-day horizon)
B. Inventory Intelligence
Stockout risk prediction
Reorder suggestions
C. Dashboard
Sales trends
Product insights
Alerts
D. Evaluation
MAE, RMSE
Latency metrics
Load testing results
5. TECHNOLOGY STACK
Backend
FastAPI (Python)
PostgreSQL
SQLAlchemy + Alembic
ML / ETL
Pandas
NumPy
Scikit-learn
XGBoost
Frontend
React / Next.js
Chart.js / Recharts
Tailwind / MUI
DevOps
Docker
GitHub
6. SYSTEM ARCHITECTURE
Raw Data → ETL Pipeline → Database → ML Models → API → Frontend Dashboard
7. DEVELOPMENT RULES (CRITICAL)
Rule 1
We follow STRICT PHASE-BASED DEVELOPMENT
👉 We DO NOT start a new phase until the previous phase is COMPLETE.
Rule 2
Each module must be:
Production-ready
Clean code
Typed (Python type hints)
Documented
Testable
Rule 3
NO placeholder or pseudo code.
Claude must generate:
Fully working code
Modular structure
Error handling
Rule 4
Every phase must produce:
Working output
Tested components
Stored artifacts
8. PROJECT PHASES (STRICT ORDER)
🔵 PHASE 1 — FOUNDATION & DATABASE
Goal
Create project structure + database schema
Tasks
Setup repository structure
Setup FastAPI project
Setup PostgreSQL connection
Define SQLAlchemy models
Configure Alembic migrations
Tables to Create
products
sales_daily
inventory_signals
forecasts
model_runs
api_logs
Deliverables
Working FastAPI server
Database connected
Tables created
Migration working
👉 ❗ DO NOT MOVE TO PHASE 2 UNTIL:
DB schema is finalized
API runs successfully
Tables tested
🟢 PHASE 2 — ETL PIPELINE
Goal
Convert raw transaction data into forecasting-ready dataset
Tasks
Step 1 — Extract
Load dataset
Standardize columns
Parse dates
Step 2 — Clean
Handle nulls
Remove invalid rows
Process returns/cancellations
Step 3 — Transform
Create revenue column
Aggregate to daily product-level
Step 4 — Feature Engineering
lag_1, lag_7, lag_14
rolling_mean_7
rolling_std_7
weekday, month
Step 5 — Load
Store processed data into DB
Step 6 — Stream Simulator
Simulate real-time ingestion
Deliverables
Clean dataset
Aggregated dataset
Feature dataset
ETL scripts modularized
👉 ❗ DO NOT MOVE TO PHASE 3 UNTIL:
ETL runs end-to-end
Data is stored in DB
Features validated
🟡 PHASE 3 — MACHINE LEARNING
Goal
Build predictive models
Tasks
Baseline Models
Naive (lag-based)
Random Forest
XGBoost
Target
Predict daily demand (regression)
Evaluation
MAE
RMSE
Validation Strategy
Time-based split (NOT random)
Output
Trained model
Metrics
Feature importance
Deliverables
Model artifact
Metrics report
Comparison results
👉 ❗ DO NOT MOVE TO PHASE 4 UNTIL:
Best model selected
Metrics documented
Model saved
🟠 PHASE 4 — BACKEND API
Goal
Serve predictions and analytics via API
Endpoints
Forecast
GET /forecast/{stock_code}
Inventory
GET /inventory/{stock_code}
Dashboard
GET /dashboard/summary
Metrics
GET /models/metrics
Health
GET /health
Deliverables
Working APIs
JSON responses
Model integrated
👉 ❗ DO NOT MOVE TO PHASE 5 UNTIL:
APIs tested via Postman
Latency measured
🔴 PHASE 5 — FRONTEND DASHBOARD
Goal
Build SaaS interface
Pages
Dashboard Overview
Product Forecast
Inventory Alerts
Model Analytics
Deliverables
Functional UI
API integration
Charts working
👉 ❗ DO NOT MOVE TO PHASE 6 UNTIL:
UI fully connected to backend
Data renders correctly
🟣 PHASE 6 — INTEGRATION & EVALUATION
Goal
Full system testing
Tasks
End-to-end testing
Latency testing (<200ms target)
Load testing
Error handling
Metrics
MAE
RMSE
API latency
Throughput
Deliverables
Evaluation report
Performance charts
9. CODE GENERATION RULES FOR CLAUDE
Whenever generating code:
ALWAYS FOLLOW:
Clean architecture
Modular design
No hardcoding
Reusable functions
Logging included
Error handling included
NEVER:
Generate incomplete code
Skip imports
Use placeholders
Ignore performance
10. HOW TO REQUEST CODE (IMPORTANT)
Use this structure:
Module: [file name]

Context:
We are building a SaaS predictive analytics system for e-commerce.

Task:
[clear description]

Requirements:
- Python 3.11
- FastAPI / Pandas / XGBoost (as needed)
- Clean architecture
- Type hints
- Docstrings
- No placeholder code

Output:
Full working code
11. FINAL SYSTEM REQUIREMENTS
The final system MUST:
✔ Run end-to-end
✔ Provide predictions via API
✔ Display results in UI
✔ Meet latency target
✔ Produce evaluation metrics
✔ Be demo-ready
12. SUCCESS CRITERIA
Project is successful if:
ETL pipeline works correctly
Model achieves reasonable MAE/RMSE
API latency <200ms
Dashboard displays real-time insights
System is modular and scalable
13. IMPORTANT FINAL RULE
We build like a production SaaS system, NOT like a student assignment.