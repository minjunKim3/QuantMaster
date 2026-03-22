# 1단계: Java 빌드
FROM maven:3.9-eclipse-temurin-17 AS build
WORKDIR /app
COPY pom.xml .
RUN mvn dependency:resolve
COPY src ./src
RUN mvn package -DskipTests

# 2단계: 실행 환경 (Java + Python)
FROM eclipse-temurin:17-jre

RUN apt-get update && apt-get install -y python3 python3-pip python3-venv && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 가상환경 + 라이브러리
RUN python3 -m venv /app/agent/venv
COPY agent/requirements.txt /app/agent/
RUN /app/agent/venv/bin/pip install -r /app/agent/requirements.txt

# Python 스크립트 + 모델 파일 복사
COPY agent/*.py /app/agent/
COPY agent/models/ /app/agent/models/

# Java JAR 복사
COPY --from=build /app/target/*.jar app.jar

EXPOSE 8080

CMD ["java", "-jar", "app.jar"]