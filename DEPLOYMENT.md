# Deployment Guide

## 🐳 Docker Deployment

### Prerequisites
- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed (usually comes with Docker Desktop)

### Quick Start with Docker

1. **Build the Docker image:**
```bash
docker-compose build
```

2. **Start the API server:**
```bash
docker-compose up -d
```

3. **Check if the service is running:**
```bash
docker-compose ps
```

4. **View logs:**
```bash
docker-compose logs -f churn-api
```

5. **Test the API:**
```bash
curl http://localhost:5000/health
```

6. **Stop the service:**
```bash
docker-compose down
```

### Manual Docker Build

If you prefer to build and run manually:

```bash
# Build image
docker build -t mlp-churn-api:latest .

# Run container
docker run -d \
  --name churn-classifier \
  -p 5000:5000 \
  -v $(pwd)/mlp_churn_classifier.pth:/app/mlp_churn_classifier.pth:ro \
  mlp-churn-api:latest

# Check logs
docker logs -f churn-classifier

# Stop container
docker stop churn-classifier
docker rm churn-classifier
```

---

## 🚀 Local Deployment (Without Docker)

### 1. Setup Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Model Artifacts

Make sure you have:
- `mlp_churn_classifier.pth` (trained model)
- Run preprocessing to create artifacts:

```bash
python -c "from app import preprocess_and_save_artifacts; preprocess_and_save_artifacts()"
```

### 4. Start Flask API

```bash
python app.py
```

The API will be available at: `http://localhost:5000`

### 5. Start Streamlit Dashboard (Optional)

In a separate terminal:

```bash
streamlit run dashboard.py
```

Dashboard will open in your browser automatically.

---

## 🧪 Testing the API

### Health Check

```bash
curl http://localhost:5000/health
```

### Get Model Info

```bash
curl http://localhost:5000/model/info
```

### Make a Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "total_orders": 15,
      "total_revenue": 2500.0,
      "avg_revenue": 166.67,
      "std_revenue": 45.2,
      "total_profit": 625.0,
      "avg_profit": 41.67,
      "avg_discount": 0.05,
      "total_quantity": 30,
      "avg_quantity": 2.0,
      "days_since_last_purchase": 45,
      "customer_lifetime_days": 365,
      "purchase_frequency": 0.041,
      "Region_encoded": 2,
      "Product_Category_encoded": 1,
      "Customer_Segment_encoded": 0,
      "Payment_Method_encoded": 1
    }
  }'
```

### Python Client Example

```python
import requests
import json

url = "http://localhost:5000/predict"

data = {
    "features": {
        "total_orders": 15,
        "total_revenue": 2500.0,
        "avg_revenue": 166.67,
        "std_revenue": 45.2,
        "total_profit": 625.0,
        "avg_profit": 41.67,
        "avg_discount": 0.05,
        "total_quantity": 30,
        "avg_quantity": 2.0,
        "days_since_last_purchase": 45,
        "customer_lifetime_days": 365,
        "purchase_frequency": 0.041,
        "Region_encoded": 2,
        "Product_Category_encoded": 1,
        "Customer_Segment_encoded": 0,
        "Payment_Method_encoded": 1
    }
}

response = requests.post(url, json=data)
result = response.json()

print(f"Churn Probability: {result['churn_probability']}")
print(f"Prediction: {result['prediction_label']}")
print(f"Risk Level: {result['risk_level']}")
```

---

## ☁️ Cloud Deployment Options

### AWS Deployment (EC2)

1. **Launch EC2 instance** (t2.medium or larger recommended)
2. **Install Docker on EC2:**
```bash
sudo yum update -y
sudo yum install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user
```

3. **Transfer files and deploy:**
```bash
scp -r . ec2-user@<instance-ip>:~/churn-classifier/
ssh ec2-user@<instance-ip>
cd ~/churn-classifier
docker-compose up -d
```

4. **Configure security group** to allow inbound traffic on port 5000

### Heroku Deployment

1. **Create `Procfile`:**
```
web: python app.py
```

2. **Deploy:**
```bash
heroku login
heroku create mlp-churn-classifier
git push heroku main
heroku open
```

### Azure Deployment (Container Instances)

```bash
# Login
az login

# Create resource group
az group create --name churn-rg --location eastus

# Create container
az container create \
  --resource-group churn-rg \
  --name churn-api \
  --image mlp-churn-api:latest \
  --dns-name-label mlp-churn \
  --ports 5000
```

---

## 📊 Production Considerations

### Monitoring

- Add logging middleware for request/response tracking
- Integrate with monitoring tools (Prometheus, Grafana)
- Set up alerting for API errors or downtime

### Scaling

- Use gunicorn for production WSGI server:
  ```bash
  gunicorn -w 4 -b 0.0.0.0:5000 app:app
  ```
- Deploy behind load balancer (nginx, AWS ALB)
- Consider Kubernetes for auto-scaling

### Security

- Enable HTTPS/SSL
- Implement API authentication (JWT, API keys)
- Rate limiting to prevent abuse
- Input validation and sanitization

### Performance

- Model serving optimization (ONNX, TorchScript)
- Caching for frequent predictions
- Batch prediction endpoint for efficiency
- Model versioning and A/B testing

---

## 🔧 Troubleshooting

### Port Already in Use

```bash
# Find process using port 5000
lsof -i :5000  # Linux/Mac
netstat -ano | findstr :5000  # Windows

# Kill the process or use different port
docker-compose down
```

### Model Not Found Error

Ensure `mlp_churn_classifier.pth` exists:
```bash
# Train model first
python train.py
```

### Preprocessing Artifacts Missing

```bash
# Generate artifacts
python -c "from app import preprocess_and_save_artifacts; preprocess_and_save_artifacts()"
```

---

## 📚 API Documentation

Once deployed, visit `http://localhost:5000/` for interactive API documentation.

### Available Endpoints

- `GET /` - API documentation homepage
- `GET /health` - Health check
- `GET /model/info` - Model information
- `GET /features` - List required features
- `POST /predict` - Single prediction
- `POST /predict_batch` - Batch predictions
- `POST /explain` - Explain prediction

---

## 🎯 Next Steps

1. **Load Testing:**
   ```bash
   # Install Apache Bench
   ab -n 1000 -c 10 http://localhost:5000/health
   ```

2. **Set up CI/CD** pipeline for automated testing and deployment

3. **Configure monitoring** and alerting

4. **Implement model versioning** system

5. **Add authentication** for production use

---

For more information, see the main [README.md](README.md) or contact the maintainer.
