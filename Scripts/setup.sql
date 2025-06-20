ALTER SESSION SET query_tag = '{"origin":"sf_sit-is","name":"tasty_bytes_e2e_ml","version":{"major":1, "minor":0},"attributes":{"is_quickstart":1, "source":"sql"}}';

-- Switch to ACCOUNTADMIN role
USE ROLE ACCOUNTADMIN;

-- Create a new role for data scientists
CREATE OR REPLACE ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST;

-- set my_user_var variable to equal the logged-in user
SET my_user_var = (SELECT  '"' || CURRENT_USER() || '"' );

-- Grant role to current user
GRANT ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST TO USER identifier($my_user_var);

-- Create a new role for feature store producer
CREATE OR REPLACE ROLE TASTYBYTESENDTOENDML_FS_PRODUCER;

-- Grant role to current user
GRANT ROLE TASTYBYTESENDTOENDML_FS_PRODUCER TO USER identifier($my_user_var);

-- Create a warehouse with specified configuration
CREATE OR REPLACE WAREHOUSE TASTYBYTESENDTOENDML_DS_WH 
    SCALING_POLICY = 'STANDARD', 
    WAREHOUSE_SIZE = 'XSMALL', 
    WAREHOUSE_TYPE = 'STANDARD', 
    AUTO_RESUME = true, 
    AUTO_SUSPEND = 60, 
    MAX_CONCURRENCY_LEVEL = 8, 
    STATEMENT_TIMEOUT_IN_SECONDS = 172800;

-- Create a database
CREATE OR REPLACE DATABASE TASTYBYTESENDTOENDML_PROD;

-- Create schemas within the database
CREATE SCHEMA TASTYBYTESENDTOENDML_PROD.RAW_CUSTOMER;
CREATE SCHEMA TASTYBYTESENDTOENDML_PROD.RAW_POS;
CREATE SCHEMA TASTYBYTESENDTOENDML_PROD.REGISTRY;
CREATE SCHEMA TASTYBYTESENDTOENDML_PROD.ANALYTICS;
CREATE SCHEMA TASTYBYTESENDTOENDML_PROD.ML;
CREATE SCHEMA TASTYBYTESENDTOENDML_PROD.HARMONIZED;

-- Create a compute pool for deployment
ALTER COMPUTE POOL IF EXISTS TASTYBYTESENDTOENDML_DEPLOY_POOL STOP ALL;
DROP COMPUTE POOL IF EXISTS TASTYBYTESENDTOENDML_DEPLOY_POOL;
CREATE COMPUTE POOL TASTYBYTESENDTOENDML_DEPLOY_POOL 
    MIN_NODES = 1, 
    MAX_NODES = 5, 
    INSTANCE_FAMILY = GPU_NV_S;

-- Grant all privileges on the deploy compute pool to the data scientist role
GRANT ALL PRIVILEGES ON COMPUTE POOL TASTYBYTESENDTOENDML_DEPLOY_POOL TO ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST;

-- Create a compute pool to run the notebook
ALTER COMPUTE POOL IF EXISTS TASTYBYTESENDTOENDML_POOL STOP ALL;
DROP COMPUTE POOL IF EXISTS TASTYBYTESENDTOENDML_POOL;
CREATE COMPUTE POOL TASTYBYTESENDTOENDML_POOL 
    MIN_NODES = 1, 
    MAX_NODES = 5, 
    INSTANCE_FAMILY = GPU_NV_S;

-- Grant all privileges on the compute pool to the data scientist role
GRANT ALL PRIVILEGES ON COMPUTE POOL TASTYBYTESENDTOENDML_POOL TO ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST;


-- Create or replace a network rule for Conda access
CREATE OR REPLACE NETWORK RULE TASTYBYTESENDTOENDML_PROD.ANALYTICS.TASTYBYTESENDTOENDML_CONDA_NETWORK_RULE
    TYPE = HOST_PORT,
    MODE = EGRESS,
    VALUE_LIST = ('conda.anaconda.org', 'pypi.org', 'pypi.python.org', 'pythonhosted.org', 'files.pythonhosted.org');

-- Grant all privileges on the Conda network rule to the data scientist role
GRANT ALL PRIVILEGES ON NETWORK RULE TASTYBYTESENDTOENDML_PROD.ANALYTICS.TASTYBYTESENDTOENDML_CONDA_NETWORK_RULE 
    TO ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST;

-- Create or replace external access integration for Conda access
CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION TASTYBYTESENDTOENDML_CONDA_ACCESS_INTEGRATION 
    ALLOWED_NETWORK_RULES = (TASTYBYTESENDTOENDML_PROD.ANALYTICS.TASTYBYTESENDTOENDML_CONDA_NETWORK_RULE), 
    ENABLED = TRUE;

-- Grant all privileges on the Conda access integration to the data scientist role
GRANT ALL PRIVILEGES ON INTEGRATION TASTYBYTESENDTOENDML_CONDA_ACCESS_INTEGRATION 
    TO ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST;

-- Create or replace a network rule to allow all outbound traffic
CREATE OR REPLACE NETWORK RULE TASTYBYTESENDTOENDML_PROD.ANALYTICS.TASTYBYTESENDTOENDML_ALLOW_ALL_NETWORK_RULE
    TYPE = HOST_PORT,
    MODE = EGRESS,
    VALUE_LIST = ('0.0.0.0:443', '0.0.0.0:80');

-- Grant all privileges on the allow-all network rule to the data scientist role
GRANT ALL PRIVILEGES ON NETWORK RULE TASTYBYTESENDTOENDML_PROD.ANALYTICS.TASTYBYTESENDTOENDML_ALLOW_ALL_NETWORK_RULE 
    TO ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST;

-- Create or replace external access integration to allow all network traffic
CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION TASTYBYTESENDTOENDML_ALLOW_ALL_ACCESS_INTEGRATION 
    ALLOWED_NETWORK_RULES = (TASTYBYTESENDTOENDML_PROD.ANALYTICS.TASTYBYTESENDTOENDML_ALLOW_ALL_NETWORK_RULE), 
    ENABLED = TRUE;

-- Grant all privileges on the allow-all access integration to the data scientist role
GRANT ALL PRIVILEGES ON INTEGRATION TASTYBYTESENDTOENDML_ALLOW_ALL_ACCESS_INTEGRATION 
    TO ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST;


-- Create or replace a CSV file format in the RAW_POS schema
CREATE OR REPLACE FILE FORMAT TASTYBYTESENDTOENDML_PROD.RAW_POS.CSV_END_TO_END_ML_FF 
    TYPE = 'csv'
    NULL_IF = ('NULL', 'null', '')
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    SKIP_HEADER = 1; 

-- Create or replace an S3 stage in the RAW_POS schema
CREATE OR REPLACE STAGE TASTYBYTESENDTOENDML_PROD.RAW_POS.S3LOAD_END_TO_END_ML
    COMMENT = 'Quickstarts S3 Stage Connection',
    URL = 's3://sfquickstarts/sfguide_getting_started_with_running_distributed_pytorch_models_on_snowflake/',
    FILE_FORMAT = TASTYBYTESENDTOENDML_PROD.RAW_POS.CSV_END_TO_END_ML_FF;

-- Warehouse size change
ALTER WAREHOUSE TASTYBYTESENDTOENDML_DS_WH set warehouse_size = 'XLARGE';
    
-- Create or replace the COUNTRY table in the RAW_POS schema
create or replace TABLE TASTYBYTESENDTOENDML_PROD.RAW_POS.COUNTRY (
	COUNTRY_ID NUMBER(18,0),
	COUNTRY VARCHAR(16777216),
	ISO_CURRENCY VARCHAR(3),
	ISO_COUNTRY VARCHAR(2),
	CITY_ID NUMBER(19,0),
	CITY VARCHAR(16777216),
	CITY_POPULATION NUMBER(38,0)
);
-- Copy data into the COUNTRY table from the S3 stage
COPY INTO TASTYBYTESENDTOENDML_PROD.RAW_POS.COUNTRY
FROM @TASTYBYTESENDTOENDML_PROD.RAW_POS.S3LOAD_END_TO_END_ML/raw_pos/country/;


-- Create or replace the MENU table in the RAW_POS schema
create or replace TABLE TASTYBYTESENDTOENDML_PROD.RAW_POS.MENU (
	MENU_ID NUMBER(19,0),
	MENU_TYPE_ID NUMBER(38,0),
	MENU_TYPE VARCHAR(16777216),
	MENU_ITEM_ID NUMBER(38,0),
	MENU_ITEM_NAME VARCHAR(16777216),
	ITEM_CATEGORY VARCHAR(16777216),
	ITEM_SUBCATEGORY VARCHAR(16777216),
	COST_OF_GOODS_USD NUMBER(38,4),
	SALE_PRICE_USD NUMBER(38,4),
	MENU_ITEM_HEALTH_METRICS_OBJ VARIANT,
	TRUCK_BRAND_NAME VARCHAR(16777216)
);

-- Copy data into the MENU table from the S3 stage
COPY INTO TASTYBYTESENDTOENDML_PROD.RAW_POS.MENU
FROM @TASTYBYTESENDTOENDML_PROD.RAW_POS.S3LOAD_END_TO_END_ML/raw_pos/menu/;


-- Create or replace the LOCATION table in the RAW_POS schema
create or replace TABLE TASTYBYTESENDTOENDML_PROD.RAW_POS.LOCATION (
	LOCATION_ID NUMBER(19,0),
	PLACEKEY VARCHAR(16777216),
	LOCATION VARCHAR(16777216),
	CITY VARCHAR(16777216),
	REGION VARCHAR(16777216),
	ISO_COUNTRY_CODE VARCHAR(16777216),
	COUNTRY VARCHAR(16777216)
);

-- Copy data into the LOCATION table from the S3 stage
COPY INTO TASTYBYTESENDTOENDML_PROD.RAW_POS.LOCATION
FROM @TASTYBYTESENDTOENDML_PROD.RAW_POS.S3LOAD_END_TO_END_ML/raw_pos/location/;


-- Create or replace the ORDER_DETAIL table in the RAW_POS schema
create or replace TABLE TASTYBYTESENDTOENDML_PROD.RAW_POS.ORDER_DETAIL (
	ORDER_DETAIL_ID NUMBER(38,0),
	ORDER_ID NUMBER(38,0),
	MENU_ITEM_ID NUMBER(38,0),
	DISCOUNT_ID VARCHAR(16777216),
	LINE_NUMBER NUMBER(38,0),
	QUANTITY NUMBER(5,0),
	UNIT_PRICE NUMBER(38,4),
	PRICE NUMBER(38,4),
	ORDER_ITEM_DISCOUNT_AMOUNT VARCHAR(16777216)
);

-- Copy data into the ORDER_DETAIL table from the S3 stage
COPY INTO TASTYBYTESENDTOENDML_PROD.RAW_POS.ORDER_DETAIL
FROM @TASTYBYTESENDTOENDML_PROD.RAW_POS.S3LOAD_END_TO_END_ML/raw_pos/order_detail/;


-- Create or replace the ORDER_HEADER table in the RAW_POS schema
create or replace TABLE TASTYBYTESENDTOENDML_PROD.RAW_POS.ORDER_HEADER (
	ORDER_ID NUMBER(38,0),
	TRUCK_ID NUMBER(38,0),
	LOCATION_ID FLOAT,
	CUSTOMER_ID NUMBER(38,0),
	DISCOUNT_ID VARCHAR(16777216),
	SHIFT_ID NUMBER(38,0),
	SHIFT_START_TIME TIME(9),
	SHIFT_END_TIME TIME(9),
	ORDER_CHANNEL VARCHAR(16777216),
	SERVED_TS VARCHAR(16777216),
	ORDER_CURRENCY VARCHAR(3),
	ORDER_AMOUNT NUMBER(38,4),
	ORDER_TAX_AMOUNT VARCHAR(16777216),
	ORDER_DISCOUNT_AMOUNT VARCHAR(16777216),
	ORDER_TOTAL NUMBER(38,4),
	ORDER_TS TIMESTAMP_NTZ(9)
);

-- Copy data into the ORDER_HEADER table from the S3 stage
COPY INTO TASTYBYTESENDTOENDML_PROD.RAW_POS.ORDER_HEADER
FROM @TASTYBYTESENDTOENDML_PROD.RAW_POS.S3LOAD_END_TO_END_ML/raw_pos/order_header/;


-- Create or replace the TRUCK table in the RAW_POS schema
create or replace TABLE TASTYBYTESENDTOENDML_PROD.RAW_POS.TRUCK (
	TRUCK_ID NUMBER(38,0),
	MENU_TYPE_ID NUMBER(38,0),
	PRIMARY_CITY VARCHAR(16777216),
	REGION VARCHAR(16777216),
	ISO_REGION VARCHAR(16777216),
	COUNTRY VARCHAR(16777216),
	ISO_COUNTRY_CODE VARCHAR(16777216),
	FRANCHISE_FLAG NUMBER(38,0),
	YEAR NUMBER(38,0),
	MAKE VARCHAR(16777216),
	MODEL VARCHAR(16777216),
	EV_FLAG NUMBER(38,0),
	FRANCHISE_ID NUMBER(38,0),
	TRUCK_OPENING_DATE DATE
);

-- Copy data into the TRUCK table from the S3 stage
COPY INTO TASTYBYTESENDTOENDML_PROD.RAW_POS.TRUCK
FROM @TASTYBYTESENDTOENDML_PROD.RAW_POS.S3LOAD_END_TO_END_ML/raw_pos/truck/;


-- Create or replace the FRANCHISE table in the RAW_POS schema
create or replace TABLE TASTYBYTESENDTOENDML_PROD.RAW_POS.FRANCHISE (
	FRANCHISE_ID NUMBER(38,0),
	FIRST_NAME VARCHAR(16777216),
	LAST_NAME VARCHAR(16777216),
	CITY VARCHAR(16777216),
	COUNTRY VARCHAR(16777216),
	E_MAIL VARCHAR(16777216),
	PHONE_NUMBER VARCHAR(16777216)
);

-- Copy data into the FRANCHISE table from the S3 stage
COPY INTO TASTYBYTESENDTOENDML_PROD.RAW_POS.FRANCHISE
FROM @TASTYBYTESENDTOENDML_PROD.RAW_POS.S3LOAD_END_TO_END_ML/raw_pos/franchise/;


-- Create or replace the CUSTOMER_LOYALTY table in the RAW_CUSTOMER schema
create or replace TABLE TASTYBYTESENDTOENDML_PROD.RAW_CUSTOMER.CUSTOMER_LOYALTY (
	CUSTOMER_ID NUMBER(38,0),
	FIRST_NAME VARCHAR(16777216),
	LAST_NAME VARCHAR(16777216),
	CITY VARCHAR(16777216),
	COUNTRY VARCHAR(16777216),
	POSTAL_CODE VARCHAR(16777216),
	PREFERRED_LANGUAGE VARCHAR(16777216),
	GENDER VARCHAR(16777216),
	FAVOURITE_BRAND VARCHAR(16777216),
	MARITAL_STATUS VARCHAR(16777216),
	CHILDREN_COUNT VARCHAR(16777216),
	SIGN_UP_DATE DATE,
	BIRTHDAY_DATE DATE,
	E_MAIL VARCHAR(16777216),
	PHONE_NUMBER VARCHAR(16777216)
);

-- Copy data into the CUSTOMER_LOYALTY table from the S3 stage
COPY INTO TASTYBYTESENDTOENDML_PROD.RAW_CUSTOMER.CUSTOMER_LOYALTY
FROM @TASTYBYTESENDTOENDML_PROD.RAW_POS.S3LOAD_END_TO_END_ML/raw_customer/customer_loyalty/;


-- Create or replace the ORDERS_V view in the HARMONIZED schema
CREATE OR REPLACE VIEW TASTYBYTESENDTOENDML_PROD.HARMONIZED.ORDERS_V AS
SELECT 
    oh.ORDER_ID,
    oh.TRUCK_ID,
    oh.ORDER_TS,
    od.ORDER_DETAIL_ID,
    od.LINE_NUMBER,
    m.TRUCK_BRAND_NAME,
    m.MENU_TYPE,
    t.PRIMARY_CITY,
    t.REGION,
    t.COUNTRY,
    t.FRANCHISE_FLAG,
    t.FRANCHISE_ID,
    f.FIRST_NAME AS FRANCHISEE_FIRST_NAME,
    f.LAST_NAME AS FRANCHISEE_LAST_NAME,
    l.LOCATION_ID,
    cl.CUSTOMER_ID,
    cl.FIRST_NAME,
    cl.LAST_NAME,
    cl.E_MAIL,
    cl.PHONE_NUMBER,
    cl.CHILDREN_COUNT,
    cl.GENDER,
    cl.MARITAL_STATUS,
    od.MENU_ITEM_ID,
    m.MENU_ITEM_NAME,
    od.QUANTITY,
    od.UNIT_PRICE,
    od.PRICE,
    oh.ORDER_AMOUNT,
    oh.ORDER_TAX_AMOUNT,
    oh.ORDER_DISCOUNT_AMOUNT,
    oh.ORDER_TOTAL
FROM TASTYBYTESENDTOENDML_PROD.RAW_POS.ORDER_DETAIL od
JOIN TASTYBYTESENDTOENDML_PROD.RAW_POS.ORDER_HEADER oh
    ON od.ORDER_ID = oh.ORDER_ID
JOIN TASTYBYTESENDTOENDML_PROD.RAW_POS.TRUCK t
    ON oh.TRUCK_ID = t.TRUCK_ID
JOIN TASTYBYTESENDTOENDML_PROD.RAW_POS.MENU m
    ON od.MENU_ITEM_ID = m.MENU_ITEM_ID
JOIN TASTYBYTESENDTOENDML_PROD.RAW_POS.FRANCHISE f
    ON t.FRANCHISE_ID = f.FRANCHISE_ID
JOIN TASTYBYTESENDTOENDML_PROD.RAW_POS.LOCATION l
    ON oh.LOCATION_ID = l.LOCATION_ID
LEFT JOIN TASTYBYTESENDTOENDML_PROD.RAW_CUSTOMER.CUSTOMER_LOYALTY cl
    ON oh.CUSTOMER_ID = cl.CUSTOMER_ID;


-- Create or replace the ORDERS_V view in the ANALYTICS schema
CREATE OR REPLACE VIEW TASTYBYTESENDTOENDML_PROD.ANALYTICS.ORDERS_V
COMMENT = 'Tasty Bytes Order Detail View' AS
SELECT DATE(o.ORDER_TS) AS DATE, * 
FROM TASTYBYTESENDTOENDML_PROD.HARMONIZED.ORDERS_V o;

-- Create or replace the LOYALTY_PURCHASED_ITEMS view in the ANALYTICS schema
CREATE OR REPLACE VIEW TASTYBYTESENDTOENDML_PROD.ANALYTICS.LOYALTY_PURCHASED_ITEMS (
    CUSTOMER_ID,
    MENU_ITEM_NAME,
    PURCHASED
) AS (
    SELECT 
        a.CUSTOMER_ID, 
        b.MENU_ITEM_NAME, 
        CASE 
            WHEN c.PURCHASED IS NOT NULL THEN 1
            ELSE 0
        END AS PURCHASED
    FROM 
        TASTYBYTESENDTOENDML_PROD.RAW_CUSTOMER.CUSTOMER_LOYALTY a
    CROSS JOIN
        (SELECT MENU_ITEM_NAME 
         FROM TASTYBYTESENDTOENDML_PROD.RAW_POS.MENU 
         WHERE ITEM_CATEGORY != 'Beverage') b
    LEFT JOIN 
        (SELECT DISTINCT CUSTOMER_ID, MENU_ITEM_NAME, TRUCK_ID, 1 AS PURCHASED 
         FROM TASTYBYTESENDTOENDML_PROD.ANALYTICS.ORDERS_V
         WHERE CUSTOMER_ID IS NOT NULL) c 
    ON a.CUSTOMER_ID = c.CUSTOMER_ID AND b.MENU_ITEM_NAME = c.MENU_ITEM_NAME
);

  

-- Create or replace the CUSTOMER_FEATURES table in the ML schema
CREATE OR REPLACE TABLE TASTYBYTESENDTOENDML_PROD.ML.CUSTOMER_FEATURES AS
SELECT 
    CUSTOMER_ID, 
    CITY, 
    COUNTRY, 
    GENDER, 
    MARITAL_STATUS, 
    BIRTHDAY_DATE, 
    DATEDIFF(YEAR, BIRTHDAY_DATE, CURRENT_DATE()) AS AGE 
FROM 
    TASTYBYTESENDTOENDML_PROD.RAW_CUSTOMER.CUSTOMER_LOYALTY;

-- Create or replace the MENU_ITEM_FEATURES table in the ML schema
CREATE OR REPLACE TABLE TASTYBYTESENDTOENDML_PROD.ML.MENU_ITEM_FEATURES AS 
SELECT 
    MENU_TYPE, 
    TRUCK_BRAND_NAME, 
    MENU_ITEM_NAME, 
    ITEM_CATEGORY, 
    ITEM_SUBCATEGORY, 
    SALE_PRICE_USD 
FROM 
    TASTYBYTESENDTOENDML_PROD.RAW_POS.MENU;

-- Revert Warehouse Size
ALTER WAREHOUSE TASTYBYTESENDTOENDML_DS_WH set warehouse_size = 'XSMALL';

-- Grant privileges on the DATABASE to the SYSADMIN and DATA SCIENTIST roles
GRANT ALL ON DATABASE TASTYBYTESENDTOENDML_PROD TO ROLE SYSADMIN;
GRANT ALL ON DATABASE TASTYBYTESENDTOENDML_PROD TO ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST;

-- Grant privileges on SCHEMAS to the SYSADMIN and DATA SCIENTIST roles
GRANT ALL ON SCHEMA TASTYBYTESENDTOENDML_PROD.RAW_POS TO ROLE SYSADMIN;
GRANT ALL ON SCHEMA TASTYBYTESENDTOENDML_PROD.RAW_POS TO ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST;

GRANT ALL ON SCHEMA TASTYBYTESENDTOENDML_PROD.RAW_CUSTOMER TO ROLE SYSADMIN;
GRANT ALL ON SCHEMA TASTYBYTESENDTOENDML_PROD.RAW_CUSTOMER TO ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST;

GRANT ALL ON SCHEMA TASTYBYTESENDTOENDML_PROD.ML TO ROLE SYSADMIN;
GRANT ALL ON SCHEMA TASTYBYTESENDTOENDML_PROD.ML TO ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST;

GRANT ALL ON SCHEMA TASTYBYTESENDTOENDML_PROD.HARMONIZED TO ROLE SYSADMIN;
GRANT ALL ON SCHEMA TASTYBYTESENDTOENDML_PROD.HARMONIZED TO ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST;

GRANT ALL ON SCHEMA TASTYBYTESENDTOENDML_PROD.ANALYTICS TO ROLE SYSADMIN;
GRANT ALL ON SCHEMA TASTYBYTESENDTOENDML_PROD.ANALYTICS TO ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST;

GRANT ALL ON SCHEMA TASTYBYTESENDTOENDML_PROD.REGISTRY TO ROLE SYSADMIN;
GRANT ALL ON SCHEMA TASTYBYTESENDTOENDML_PROD.REGISTRY TO ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST;

-- Grant privileges on ALL TABLES in SCHEMAS to the DATA SCIENTIST role
GRANT ALL ON ALL TABLES IN SCHEMA TASTYBYTESENDTOENDML_PROD.RAW_POS TO ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST;
GRANT ALL ON ALL TABLES IN SCHEMA TASTYBYTESENDTOENDML_PROD.RAW_CUSTOMER TO ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST;
GRANT ALL ON ALL TABLES IN SCHEMA TASTYBYTESENDTOENDML_PROD.ML TO ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST;

-- Grant privileges on ALL VIEWS in SCHEMAS to the DATA SCIENTIST role
GRANT ALL ON ALL VIEWS IN SCHEMA TASTYBYTESENDTOENDML_PROD.HARMONIZED TO ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST;
GRANT ALL ON ALL VIEWS IN SCHEMA TASTYBYTESENDTOENDML_PROD.ANALYTICS TO ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST;

-- Grant privileges on the WAREHOUSE to the SYSADMIN and DATA SCIENTIST roles
GRANT ALL ON WAREHOUSE TASTYBYTESENDTOENDML_DS_WH TO ROLE SYSADMIN;
GRANT ALL ON WAREHOUSE TASTYBYTESENDTOENDML_DS_WH TO ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST;

-- Create the ML_STAGE in the ML schema if it does not exist
CREATE OR REPLACE STAGE TASTYBYTESENDTOENDML_PROD.ML.ML_STAGE 
DIRECTORY = (ENABLE = TRUE) 
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
GRANT READ, WRITE ON STAGE TASTYBYTESENDTOENDML_PROD.ML.ML_STAGE TO ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST;

-- Create the IMAGE_REPOSITORY in the registry schema if it does not exist
CREATE OR REPLACE IMAGE REPOSITORY TASTYBYTESENDTOENDML_PROD.REGISTRY.IMAGE_REPO;
GRANT OWNERSHIP ON IMAGE REPOSITORY TASTYBYTESENDTOENDML_PROD.REGISTRY.IMAGE_REPO TO ROLE TASTYBYTESENDTOENDML_DATA_SCIENTIST COPY CURRENT GRANTS;
