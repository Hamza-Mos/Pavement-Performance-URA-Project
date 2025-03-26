-- Pavement Monitoring System Database Schema
-- Optimized for Time Series Analysis
-- Compatible with TimescaleDB, InfluxDB, and other time series databases

-- Sensor Information Table
-- Contains metadata about all sensors in the system
CREATE TABLE SensorInformation (
    SensorID VARCHAR(20) PRIMARY KEY,
    SensorType VARCHAR(20) NOT NULL CHECK (SensorType IN ('TEMPERATURE', 'STRAIN', 'PRESSURE', 'MOISTURE', 'WEATHER')),
    ModelNumber VARCHAR(50),
    Orientation VARCHAR(20) CHECK (Orientation IN ('LONGITUDINAL', 'TRANSVERSE', 'VERTICAL', 'N/A')),
    WheelPath VARCHAR(10) CHECK (WheelPath IN ('RIGHT', 'LEFT', 'N/A')),
    Depth DECIMAL(6,1) NOT NULL,
    LayerType VARCHAR(20) NOT NULL CHECK (LayerType IN ('ASPHALT', 'BASE', 'SUBBASE', 'SUBGRADE', 'AIR', 'N/A')),
    XCoordinate DECIMAL(9,3),
    YCoordinate DECIMAL(9,3),
    ZCoordinate DECIMAL(9,3),
    InstallationDate TIMESTAMP,
    CalibrationFormula TEXT,
    MeasurementUnit VARCHAR(10) NOT NULL,
    MinValue DECIMAL(9,3),
    MaxValue DECIMAL(9,3),
    Accuracy DECIMAL(9,3),
    Description TEXT,
    IsActive BOOLEAN DEFAULT TRUE,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UpdatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Project Information Table
-- Contains metadata about the overall project
CREATE TABLE ProjectInformation (
    ProjectID VARCHAR(20) PRIMARY KEY,
    ProjectName VARCHAR(100) NOT NULL,
    Location VARCHAR(100),
    Latitude DECIMAL(9,6),
    Longitude DECIMAL(9,6),
    PavementAge DECIMAL(5,2),
    ConstructionDate TIMESTAMP,
    PavementType VARCHAR(50),
    ClimateRegion VARCHAR(50),
    AnnualFreezingIndex DECIMAL(9,2),
    AnnualPrecipitation DECIMAL(9,2),
    AnnualAverageTemperature DECIMAL(5,2),
    AADT INTEGER,
    AADTT INTEGER,
    Description TEXT,
    ContactInformation TEXT,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UpdatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Raw Temperature Data Table (Time Series)
-- Optimized for high-frequency measurements
CREATE TABLE RawTemperatureData (
    ReadingID BIGINT PRIMARY KEY AUTOINCREMENT,
    Timestamp TIMESTAMP NOT NULL,  -- Primary time dimension
    SensorID VARCHAR(20) NOT NULL,
    TemperatureValue DECIMAL(5,2) NOT NULL,
    FOREIGN KEY (SensorID) REFERENCES SensorInformation(SensorID)
);

-- Create hypertable (for TimescaleDB) or similar optimization
-- CREATE HYPERTABLE RawTemperatureData(Timestamp);
CREATE INDEX idx_temp_timestamp ON RawTemperatureData(Timestamp DESC);
CREATE INDEX idx_temp_sensor_time ON RawTemperatureData(SensorID, Timestamp DESC);

-- Weather Data Table (Time Series)
-- Stores raw weather readings
CREATE TABLE WeatherData (
    WeatherID BIGINT PRIMARY KEY AUTOINCREMENT,
    Timestamp TIMESTAMP NOT NULL,  -- Primary time dimension
    AirTemperature DECIMAL(5,2),
    RelativeHumidity DECIMAL(5,2),
    Precipitation DECIMAL(6,2),
    WindSpeed DECIMAL(5,2),
    WindDirection VARCHAR(10),
    BarometricPressure DECIMAL(6,2),
    SolarRadiation DECIMAL(7,2),
    DataSource VARCHAR(50),
    RecordedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- CREATE HYPERTABLE WeatherData(Timestamp);
CREATE INDEX idx_weather_timestamp ON WeatherData(Timestamp DESC);

-- Strain Data Table (Time Series)
CREATE TABLE StrainData (
    ReadingID BIGINT PRIMARY KEY AUTOINCREMENT,
    Timestamp TIMESTAMP NOT NULL,  -- Primary time dimension
    SensorID VARCHAR(20) NOT NULL,
    StrainValue DECIMAL(9,3) NOT NULL,
    FOREIGN KEY (SensorID) REFERENCES SensorInformation(SensorID)
);

-- CREATE HYPERTABLE StrainData(Timestamp);
CREATE INDEX idx_strain_timestamp ON StrainData(Timestamp DESC);
CREATE INDEX idx_strain_sensor_time ON StrainData(SensorID, Timestamp DESC);

-- Pressure Data Table (Time Series)
CREATE TABLE PressureData (
    ReadingID BIGINT PRIMARY KEY AUTOINCREMENT,
    Timestamp TIMESTAMP NOT NULL,  -- Primary time dimension
    SensorID VARCHAR(20) NOT NULL,
    PressureValue DECIMAL(9,3) NOT NULL,
    FOREIGN KEY (SensorID) REFERENCES SensorInformation(SensorID)
);

-- CREATE HYPERTABLE PressureData(Timestamp);
CREATE INDEX idx_pressure_timestamp ON PressureData(Timestamp DESC);
CREATE INDEX idx_pressure_sensor_time ON PressureData(SensorID, Timestamp DESC);

-- Moisture Data Table (Time Series)
CREATE TABLE MoistureData (
    ReadingID BIGINT PRIMARY KEY AUTOINCREMENT,
    Timestamp TIMESTAMP NOT NULL,  -- Primary time dimension
    SensorID VARCHAR(20) NOT NULL,
    MoistureValue DECIMAL(9,3) NOT NULL,
    RawValue DECIMAL(9,3),
    FOREIGN KEY (SensorID) REFERENCES SensorInformation(SensorID)
);

-- CREATE HYPERTABLE MoistureData(Timestamp);
CREATE INDEX idx_moisture_timestamp ON MoistureData(Timestamp DESC);
CREATE INDEX idx_moisture_sensor_time ON MoistureData(SensorID, Timestamp DESC);

-- 15-Minute Aggregated Temperature Data (Time Series)
CREATE TABLE TempData15Min (
    AggregateID BIGINT PRIMARY KEY AUTOINCREMENT,
    Timestamp TIMESTAMP NOT NULL,  -- Start of 15-minute interval
    SensorID VARCHAR(20) NOT NULL,
    MeanValue DECIMAL(5,2) NOT NULL,
    MinValue DECIMAL(5,2) NOT NULL,
    MaxValue DECIMAL(5,2) NOT NULL,
    StdDevValue DECIMAL(5,2) NOT NULL,
    ReadingCount INTEGER NOT NULL,
    IntervalEnd TIMESTAMP NOT NULL,  -- End of 15-minute interval
    ProcessedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (SensorID) REFERENCES SensorInformation(SensorID),
    UNIQUE (Timestamp, SensorID)
);

-- CREATE HYPERTABLE TempData15Min(Timestamp);
CREATE INDEX idx_temp15min_timestamp ON TempData15Min(Timestamp DESC);
CREATE INDEX idx_temp15min_sensor_time ON TempData15Min(SensorID, Timestamp DESC);

-- Hourly Aggregated Temperature Data (Time Series)
CREATE TABLE TempDataHourly (
    AggregateID BIGINT PRIMARY KEY AUTOINCREMENT,
    Timestamp TIMESTAMP NOT NULL,  -- Start of hourly interval
    SensorID VARCHAR(20) NOT NULL,
    MeanValue DECIMAL(5,2) NOT NULL,
    MinValue DECIMAL(5,2) NOT NULL,
    MaxValue DECIMAL(5,2) NOT NULL,
    StdDevValue DECIMAL(5,2) NOT NULL,
    ReadingCount INTEGER NOT NULL,
    IntervalEnd TIMESTAMP NOT NULL,  -- End of hourly interval
    ProcessedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (SensorID) REFERENCES SensorInformation(SensorID),
    UNIQUE (Timestamp, SensorID)
);

-- CREATE HYPERTABLE TempDataHourly(Timestamp);
CREATE INDEX idx_temphourly_timestamp ON TempDataHourly(Timestamp DESC);
CREATE INDEX idx_temphourly_sensor_time ON TempDataHourly(SensorID, Timestamp DESC);

-- Daily Aggregated Temperature Data (Time Series)
CREATE TABLE TempDataDaily (
    AggregateID BIGINT PRIMARY KEY AUTOINCREMENT,
    Timestamp TIMESTAMP NOT NULL,  -- Start of day (midnight)
    SensorID VARCHAR(20) NOT NULL,
    MeanValue DECIMAL(5,2) NOT NULL,
    MinValue DECIMAL(5,2) NOT NULL,
    MaxValue DECIMAL(5,2) NOT NULL,
    StdDevValue DECIMAL(5,2) NOT NULL,
    TempRange DECIMAL(5,2) NOT NULL,
    ReadingCount INTEGER NOT NULL,
    MinTimestamp TIMESTAMP,        -- Time of day when min temp occurred
    MaxTimestamp TIMESTAMP,        -- Time of day when max temp occurred
    DayNightDelta DECIMAL(5,2),    -- Difference between day and night temps
    IntervalEnd TIMESTAMP NOT NULL, -- End of day
    ProcessedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (SensorID) REFERENCES SensorInformation(SensorID),
    UNIQUE (Timestamp, SensorID)
);

-- CREATE HYPERTABLE TempDataDaily(Timestamp);
CREATE INDEX idx_tempdaily_timestamp ON TempDataDaily(Timestamp DESC);
CREATE INDEX idx_tempdaily_sensor_time ON TempDataDaily(SensorID, Timestamp DESC);

-- Weekly Aggregated Temperature Data (Time Series)
CREATE TABLE TempDataWeekly (
    AggregateID BIGINT PRIMARY KEY AUTOINCREMENT,
    Timestamp TIMESTAMP NOT NULL,  -- Start of week (Monday)
    SensorID VARCHAR(20) NOT NULL,
    MeanValue DECIMAL(5,2) NOT NULL,
    MinValue DECIMAL(5,2) NOT NULL,
    MaxValue DECIMAL(5,2) NOT NULL,
    StdDevValue DECIMAL(5,2) NOT NULL,
    TempRange DECIMAL(5,2) NOT NULL,
    ReadingCount INTEGER NOT NULL,
    LayerType VARCHAR(20) NOT NULL,
    Depth DECIMAL(6,1) NOT NULL,
    MinTimestamp TIMESTAMP,        -- Time during week when min temp occurred
    MaxTimestamp TIMESTAMP,        -- Time during week when max temp occurred
    WeekdayAvgTemp DECIMAL(5,2),   -- Average temperature on weekdays
    WeekendAvgTemp DECIMAL(5,2),   -- Average temperature on weekends
    IntervalEnd TIMESTAMP NOT NULL, -- End of week (Sunday)
    ProcessedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (SensorID) REFERENCES SensorInformation(SensorID),
    UNIQUE (Timestamp, SensorID)
);

-- CREATE HYPERTABLE TempDataWeekly(Timestamp);
CREATE INDEX idx_tempweekly_timestamp ON TempDataWeekly(Timestamp DESC);
CREATE INDEX idx_tempweekly_sensor_time ON TempDataWeekly(SensorID, Timestamp DESC);

-- Hourly Aggregated Weather Data (Time Series)
CREATE TABLE WeatherDataHourly (
    AggregateID BIGINT PRIMARY KEY AUTOINCREMENT,
    Timestamp TIMESTAMP NOT NULL,  -- Start of hourly interval
    MeanAirTemp DECIMAL(5,2),
    MinAirTemp DECIMAL(5,2),
    MaxAirTemp DECIMAL(5,2),
    StdDevAirTemp DECIMAL(5,2),
    MeanHumidity DECIMAL(5,2),
    TotalPrecipitation DECIMAL(6,2),
    MeanWindSpeed DECIMAL(5,2),
    MaxWindSpeed DECIMAL(5,2),
    PredominantWindDirection VARCHAR(10),
    MeanBarometricPressure DECIMAL(6,2),
    MeanSolarRadiation DECIMAL(7,2),
    ReadingCount INTEGER NOT NULL,
    IntervalEnd TIMESTAMP NOT NULL, -- End of hourly interval
    ProcessedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (Timestamp)
);

-- CREATE HYPERTABLE WeatherDataHourly(Timestamp);
CREATE INDEX idx_weatherhourly_timestamp ON WeatherDataHourly(Timestamp DESC);

-- Daily Aggregated Weather Data (Time Series)
CREATE TABLE WeatherDataDaily (
    AggregateID BIGINT PRIMARY KEY AUTOINCREMENT,
    Timestamp TIMESTAMP NOT NULL,  -- Start of day (midnight)
    MeanAirTemp DECIMAL(5,2),
    MinAirTemp DECIMAL(5,2),
    MaxAirTemp DECIMAL(5,2),
    StdDevAirTemp DECIMAL(5,2),
    AirTempRange DECIMAL(5,2),
    MeanHumidity DECIMAL(5,2),
    MinHumidity DECIMAL(5,2),
    MaxHumidity DECIMAL(5,2),
    TotalPrecipitation DECIMAL(6,2),
    MeanWindSpeed DECIMAL(5,2),
    MaxWindSpeed DECIMAL(5,2),
    PredominantWindDirection VARCHAR(10),
    MeanBarometricPressure DECIMAL(6,2),
    MeanSolarRadiation DECIMAL(7,2),
    PrecipitationHours INTEGER,    -- Hours with precipitation
    ReadingCount INTEGER NOT NULL,
    IntervalEnd TIMESTAMP NOT NULL, -- End of day
    ProcessedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (Timestamp)
);

-- CREATE HYPERTABLE WeatherDataDaily(Timestamp);
CREATE INDEX idx_weatherdaily_timestamp ON WeatherDataDaily(Timestamp DESC);

-- Weekly Aggregated Weather Data (Time Series)
CREATE TABLE WeatherDataWeekly (
    AggregateID BIGINT PRIMARY KEY AUTOINCREMENT,
    Timestamp TIMESTAMP NOT NULL,  -- Start of week (Monday)
    MeanAirTemp DECIMAL(5,2),
    MinAirTemp DECIMAL(5,2),
    MaxAirTemp DECIMAL(5,2),
    StdDevAirTemp DECIMAL(5,2),
    AirTempRange DECIMAL(5,2),
    MeanHumidity DECIMAL(5,2),
    MinHumidity DECIMAL(5,2),
    MaxHumidity DECIMAL(5,2),
    TotalPrecipitation DECIMAL(6,2),
    MeanWindSpeed DECIMAL(5,2),
    MaxWindSpeed DECIMAL(5,2),
    PredominantWindDirection VARCHAR(10),
    MeanBarometricPressure DECIMAL(6,2),
    MeanSolarRadiation DECIMAL(7,2),
    RainyDays INTEGER,             -- Days with precipitation > 0.5mm
    SunnyHours DECIMAL(5,1),       -- Total hours with significant solar radiation
    ReadingCount INTEGER NOT NULL,
    IntervalEnd TIMESTAMP NOT NULL, -- End of week (Sunday)
    ProcessedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (Timestamp)
);

-- CREATE HYPERTABLE WeatherDataWeekly(Timestamp);
CREATE INDEX idx_weatherweekly_timestamp ON WeatherDataWeekly(Timestamp DESC);

-- Temperature Gradient Analysis (Time Series)
-- For analyzing temperature variations across pavement layers
CREATE TABLE TemperatureGradientWeekly (
    GradientID BIGINT PRIMARY KEY AUTOINCREMENT,
    Timestamp TIMESTAMP NOT NULL,  -- Start of week (Monday)
    AsphaltMeanTemp DECIMAL(5,2),
    BaseMeanTemp DECIMAL(5,2), 
    SubbaseMeanTemp DECIMAL(5,2),
    SubgradeMeanTemp DECIMAL(5,2),
    AsphaltToBaseGradient DECIMAL(6,3),      -- °C/m
    BaseToSubbaseGradient DECIMAL(6,3),      -- °C/m
    SubbaseToSubgradeGradient DECIMAL(6,3),  -- °C/m
    SurfaceToBottomGradient DECIMAL(6,3),    -- °C/m
    MaxDailyGradient DECIMAL(6,3),           -- Maximum daily gradient that week
    MaxGradientTimestamp TIMESTAMP,          -- When max gradient occurred
    DiurnalVariationSurface DECIMAL(5,2),    -- Average day/night temperature difference at surface
    DiurnalVariationBase DECIMAL(5,2),       -- Average day/night temperature difference at base
    IntervalEnd TIMESTAMP NOT NULL,          -- End of week (Sunday)
    ProcessedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (Timestamp)
);

-- CREATE HYPERTABLE TemperatureGradientWeekly(Timestamp);
CREATE INDEX idx_tempgradient_timestamp ON TemperatureGradientWeekly(Timestamp DESC);

-- System Status Table
-- For tracking the status of various processing jobs
CREATE TABLE SystemStatus (
    StatusID INTEGER PRIMARY KEY AUTOINCREMENT,
    StatusType VARCHAR(50) NOT NULL,  -- e.g., '15min_aggregation', 'hourly_aggregation', etc.
    LastRunTimestamp TIMESTAMP,
    LastProcessedTimestamp TIMESTAMP,
    Status VARCHAR(20) CHECK (Status IN ('SUCCESS', 'FAILED', 'RUNNING')),
    Message TEXT,
    UpdatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (StatusType)
);

CREATE INDEX idx_status_type ON SystemStatus(StatusType);

-- Archive Information Table
-- For tracking data archive operations
CREATE TABLE ArchiveInfo (
    ArchiveID INTEGER PRIMARY KEY AUTOINCREMENT,
    IntervalStart TIMESTAMP NOT NULL,
    IntervalEnd TIMESTAMP NOT NULL,
    ArchiveDate TIMESTAMP NOT NULL,
    ArchiveFilePath VARCHAR(255) NOT NULL,
    FileSize BIGINT,
    ArchiveStatus VARCHAR(20) DEFAULT 'COMPLETED' CHECK (ArchiveStatus IN ('COMPLETED', 'FAILED', 'PARTIAL')),
    Notes TEXT,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_archive_interval ON ArchiveInfo(IntervalStart, IntervalEnd);

-- Sensor Health Monitoring (Time Series)
-- For tracking sensor performance and potential issues
CREATE TABLE SensorHealthMonitoring (
    HealthID BIGINT PRIMARY KEY AUTOINCREMENT,
    Timestamp TIMESTAMP NOT NULL,  -- Day of health check
    SensorID VARCHAR(20) NOT NULL,
    DataCaptureRate DECIMAL(5,2),  -- Percentage of expected readings received
    NoiseLevel DECIMAL(5,2),       -- Measure of signal noise
    AnomalyCount INTEGER,          -- Number of anomalous readings
    BatteryLevel DECIMAL(5,2),     -- If applicable
    SignalStrength DECIMAL(5,2),   -- If applicable
    FailureFlags INTEGER,          -- Bit flags for various failure modes
    HealthStatus VARCHAR(20) CHECK (HealthStatus IN ('GOOD', 'WARNING', 'CRITICAL', 'OFFLINE')),
    Notes TEXT,
    ProcessedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (SensorID) REFERENCES SensorInformation(SensorID),
    UNIQUE (Timestamp, SensorID)
);

-- CREATE HYPERTABLE SensorHealthMonitoring(Timestamp);
CREATE INDEX idx_sensorhealth_timestamp ON SensorHealthMonitoring(Timestamp DESC);
CREATE INDEX idx_sensorhealth_sensor_time ON SensorHealthMonitoring(SensorID, Timestamp DESC);
CREATE INDEX idx_sensorhealth_status ON SensorHealthMonitoring(HealthStatus);

-- Time Series Analysis Views

-- View: Latest sensor readings
CREATE VIEW LatestSensorReadings AS
SELECT s.SensorID, s.SensorType, s.LayerType, s.Depth, 
       CASE 
         WHEN s.SensorType = 'TEMPERATURE' THEN (SELECT TemperatureValue FROM RawTemperatureData WHERE SensorID = s.SensorID ORDER BY Timestamp DESC LIMIT 1)
         WHEN s.SensorType = 'PRESSURE' THEN (SELECT PressureValue FROM PressureData WHERE SensorID = s.SensorID ORDER BY Timestamp DESC LIMIT 1)
         WHEN s.SensorType = 'STRAIN' THEN (SELECT StrainValue FROM StrainData WHERE SensorID = s.SensorID ORDER BY Timestamp DESC LIMIT 1)
         WHEN s.SensorType = 'MOISTURE' THEN (SELECT MoistureValue FROM MoistureData WHERE SensorID = s.SensorID ORDER BY Timestamp DESC LIMIT 1)
       END AS LatestValue,
       CASE 
         WHEN s.SensorType = 'TEMPERATURE' THEN (SELECT Timestamp FROM RawTemperatureData WHERE SensorID = s.SensorID ORDER BY Timestamp DESC LIMIT 1)
         WHEN s.SensorType = 'PRESSURE' THEN (SELECT Timestamp FROM PressureData WHERE SensorID = s.SensorID ORDER BY Timestamp DESC LIMIT 1)
         WHEN s.SensorType = 'STRAIN' THEN (SELECT Timestamp FROM StrainData WHERE SensorID = s.SensorID ORDER BY Timestamp DESC LIMIT 1)
         WHEN s.SensorType = 'MOISTURE' THEN (SELECT Timestamp FROM MoistureData WHERE SensorID = s.SensorID ORDER BY Timestamp DESC LIMIT 1)
       END AS LatestTimestamp,
       s.MeasurementUnit
FROM SensorInformation s
WHERE s.IsActive = TRUE;

-- View: Current temperature profile by depth
CREATE VIEW CurrentTemperatureProfile AS
SELECT s.Depth, s.LayerType, t.MeanValue as Temperature, t.Timestamp
FROM TempDataHourly t
JOIN SensorInformation s ON t.SensorID = s.SensorID
WHERE s.SensorType = 'TEMPERATURE'
AND t.Timestamp = (SELECT MAX(Timestamp) FROM TempDataHourly)
ORDER BY s.Depth;

-- View: Weekly temperature trends
CREATE VIEW WeeklyTemperatureTrends AS
SELECT 
    w.Timestamp as WeekStart,
    w.IntervalEnd as WeekEnd,
    w.MeanAirTemp as AirTemperature,
    asphalt.MeanValue as AsphaltTemperature,
    base.MeanValue as BaseTemperature,
    subbase.MeanValue as SubbaseTemperature,
    subgrade.MeanValue as SubgradeTemperature,
    w.TotalPrecipitation,
    w.MeanHumidity
FROM WeatherDataWeekly w
LEFT JOIN (
    SELECT Timestamp, AVG(MeanValue) as MeanValue
    FROM TempDataWeekly t
    JOIN SensorInformation s ON t.SensorID = s.SensorID
    WHERE s.LayerType = 'ASPHALT'
    GROUP BY Timestamp
) asphalt ON w.Timestamp = asphalt.Timestamp
LEFT JOIN (
    SELECT Timestamp, AVG(MeanValue) as MeanValue
    FROM TempDataWeekly t
    JOIN SensorInformation s ON t.SensorID = s.SensorID
    WHERE s.LayerType = 'BASE'
    GROUP BY Timestamp
) base ON w.Timestamp = base.Timestamp
LEFT JOIN (
    SELECT Timestamp, AVG(MeanValue) as MeanValue
    FROM TempDataWeekly t
    JOIN SensorInformation s ON t.SensorID = s.SensorID
    WHERE s.LayerType = 'SUBBASE'
    GROUP BY Timestamp
) subbase ON w.Timestamp = subbase.Timestamp
LEFT JOIN (
    SELECT Timestamp, AVG(MeanValue) as MeanValue
    FROM TempDataWeekly t
    JOIN SensorInformation s ON t.SensorID = s.SensorID
    WHERE s.LayerType = 'SUBGRADE'
    GROUP BY Timestamp
) subgrade ON w.Timestamp = subgrade.Timestamp
ORDER BY w.Timestamp DESC;

-- Sample time-series queries

-- 1. Get temperature changes over time for a specific sensor and time range
-- SELECT Timestamp, TemperatureValue 
-- FROM RawTemperatureData
-- WHERE SensorID = 'TEMPS-02-1' 
--   AND Timestamp BETWEEN '2023-01-01' AND '2023-01-07'
-- ORDER BY Timestamp;

-- 2. Calculate the rate of temperature change
-- SELECT 
--   Timestamp,
--   TemperatureValue,
--   (TemperatureValue - LAG(TemperatureValue) OVER (ORDER BY Timestamp)) AS temp_change,
--   (EXTRACT(EPOCH FROM (Timestamp - LAG(Timestamp) OVER (ORDER BY Timestamp)))) AS seconds_diff,
--   (TemperatureValue - LAG(TemperatureValue) OVER (ORDER BY Timestamp)) / 
--     NULLIF((EXTRACT(EPOCH FROM (Timestamp - LAG(Timestamp) OVER (ORDER BY Timestamp)))), 0) * 3600 AS change_per_hour
-- FROM RawTemperatureData
-- WHERE SensorID = 'TEMPS-02-1' 
--   AND Timestamp BETWEEN '2023-01-01' AND '2023-01-02'
-- ORDER BY Timestamp;

-- 3. Compare temperature at different depths for the same time period
-- SELECT 
--   t.Timestamp,
--   MAX(CASE WHEN s.LayerType = 'ASPHALT' THEN t.MeanValue END) AS AsphaltTemp,
--   MAX(CASE WHEN s.LayerType = 'BASE' THEN t.MeanValue END) AS BaseTemp,
--   MAX(CASE WHEN s.LayerType = 'SUBBASE' THEN t.MeanValue END) AS SubbaseTemp,
--   MAX(CASE WHEN s.LayerType = 'SUBGRADE' THEN t.MeanValue END) AS SubgradeTemp
-- FROM TempDataHourly t
-- JOIN SensorInformation s ON t.SensorID = s.SensorID
-- WHERE t.Timestamp BETWEEN '2023-01-01' AND '2023-01-07'
-- GROUP BY t.Timestamp
-- ORDER BY t.Timestamp;

-- 4. Find temperature anomalies (values outside normal range)
-- WITH sensor_stats AS (
--   SELECT 
--     SensorID,
--     AVG(TemperatureValue) AS avg_temp,
--     STDDEV(TemperatureValue) AS std_temp
--   FROM RawTemperatureData
--   GROUP BY SensorID
-- )
-- SELECT 
--   r.Timestamp,
--   r.SensorID,
--   r.TemperatureValue,
--   s.avg_temp,
--   s.std_temp,
--   (r.TemperatureValue - s.avg_temp) / s.std_temp AS z_score
-- FROM RawTemperatureData r
-- JOIN sensor_stats s ON r.SensorID = s.SensorID
-- WHERE ABS((r.TemperatureValue - s.avg_temp) / s.std_temp) > 3
-- ORDER BY ABS((r.TemperatureValue - s.avg_temp) / s.std_temp) DESC;

-- Note: Comments for TimescaleDB specific functionality
-- To enable TimescaleDB hypertables, uncomment the CREATE HYPERTABLE lines
-- For InfluxDB, different syntax would be needed in an InfluxDB-specific script