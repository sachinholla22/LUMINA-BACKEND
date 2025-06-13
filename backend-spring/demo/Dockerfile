# Stage 1: Build the application
FROM maven:3.9.6-eclipse-temurin-17 AS build

WORKDIR /app

# Copy Maven project files
COPY pom.xml .
COPY src ./src

# Package the application
RUN mvn clean package -DskipTests



# Stage 2: Run the application
FROM openjdk:17-jdk-slim

WORKDIR /app


# Copy the built jar from the build stage
COPY --from=build /app/target/demo-0.0.1-SNAPSHOT.jar springapp.jar

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "springapp.jar"]