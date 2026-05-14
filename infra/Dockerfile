FROM eclipse-temurin:17-jre-jammy

# 1. Sistem paketleri (Python ve Build araçları)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

ENV JAVA_HOME=/opt/java/openjdk
ENV PATH=$PATH:$JAVA_HOME/bin
ENV PYSPARK_SUBMIT_ARGS="--conf spark.driver.extraJavaOptions='--add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED' pyspark-shell"

WORKDIR /app

# 🚀 ÖNEMLİ: Sabit olan JAR indirme işlemini Pip kurulumundan ÖNCE yapıyoruz
# Bu dosya internetten bir kez iner ve requirements değişse bile tekrar inmez
RUN mkdir -p /app/jars && \
    python3 -c "import urllib.request; \
    urllib.request.urlretrieve('https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.3/spark-sql-kafka-0-10_2.12-3.5.3.jar', \
    '/app/jars/spark-sql-kafka-0-10_2.12-3.5.3.jar')"

# 🚀 ÖNEMLİ 2: Kütüphane kurulumu
COPY requirements.txt .

# Mount cache kullanarak yerel önbelleği de kullanabiliriz
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# 🚀 ÖNEMLİ 3: Kod kopyalama (En son adım)
# src/api/main.py içinde bir harf değiştirsen bile yukarıdaki PIP ve JAR adımları çalışmaz!
COPY . .

RUN find / -name "log4j.properties" 2>/dev/null | head -1 | xargs -I{} sh -c \
    'echo "log4j.logger.org.apache.spark.sql.execution.window.WindowExec=ERROR" >> {}'

ENV PYTHONPATH=/app

CMD ["python", "src/api/main.py"]