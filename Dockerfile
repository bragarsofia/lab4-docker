# Модель: Оптимальне керування процесом очищення водойми
# Автор: Брагар Софія в групі з Маляренко Анастасією, група АІ-233

FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy scipy
COPY main.py .
CMD ["python", "main.py"]
