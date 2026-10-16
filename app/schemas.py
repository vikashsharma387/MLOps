from pydantic import BaseModel, Field


class HeartDiseaseInput(BaseModel):
    """
    Input schema for Heart Disease Risk Prediction.
    Validates all 13 clinical features.
    """
    age: int = Field(..., ge=1, le=120, description="Age in years", json_schema_extra={"example": 63})
    sex: int = Field(..., ge=0, le=1, description="Sex (1=male; 0=female)", json_schema_extra={"example": 1})
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)", json_schema_extra={"example": 3})
    trestbps: int = Field(..., ge=50, le=300, description="Resting BP (mm Hg)", json_schema_extra={"example": 145})
    chol: int = Field(..., ge=50, le=600, description="Cholesterol (mg/dl)", json_schema_extra={"example": 233})
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120", json_schema_extra={"example": 1})
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)", json_schema_extra={"example": 0})
    thalach: int = Field(..., ge=50, le=250, description="Max heart rate", json_schema_extra={"example": 150})
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1/0)", json_schema_extra={"example": 0})
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression", json_schema_extra={"example": 2.3})
    slope: int = Field(..., ge=0, le=2, description="Slope ST segment (0-2)", json_schema_extra={"example": 0})
    ca: int = Field(..., ge=0, le=4, description="Vessels colored (0-4)", json_schema_extra={"example": 0})
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (1-3)", json_schema_extra={"example": 1})


class PredictionResponse(BaseModel):
    """Output schema for model prediction response."""
    prediction: int = Field(..., description="Binary prediction: 1 = Disease Risk, 0 = No Disease")
    confidence: float = Field(..., description="Model prediction confidence score [0.0 - 1.0]")
    probability_disease: float = Field(..., description="Probability of heart disease")
    risk_level: str = Field(..., description="Risk assessment: HIGH RISK or LOW RISK")
    status: str = Field(default="success")
