docker build -t katonic/apps:loan-default-prediction .
docker push katonic/apps:loan-default-prediction
# docker run --rm -p 8050:8050 katonic/apps:loan-default-prediction