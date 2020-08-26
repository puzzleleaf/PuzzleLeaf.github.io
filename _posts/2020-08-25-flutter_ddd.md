---
title: "Flutter DDD"
categories: Flutter
tags: Flutter
toc: true  
toc_sticky: true 
---
아키텍처, 모든 종류의 소프트웨어에서 매우 유익하다.
적절한 아키텍처가 없으면 코드가 분리되기 시작한다.
테스트할수 없으며 점점 엉망이 될것입니다.

# Domain-Driven Design
도메인 계층은 다른 모든 계층과 완전히 독립적인 계층이다.    
(순수 비즈니스 로직 및 데이터만 포함)

프레젠테이션
- 일반적인 플러터, 위젯

어플리케이션
- BLoCs 용어를 사용하고 있다.
- 비즈니스 로직이 없다.
- 오직 데이터의 흐름을 조정하는 역할을 수행

도메인
- 앱의 필수 레이어 및 가중 중앙레이어 (엔티티를 포함함)
- 엔티티
  - 값이 유효한지 로직을 포함

인프라 계층
- API 및 데이터베이스 및 디바이스센서의 경계이 있다.
- 매우 빠르고 엄청나게 변경될 수 있다.
- Repositories - 저장소 객체에서 DataSource를 다룬다.
- Data Sources


# Referece
- https://resocoder.com/2020/03/09/flutter-firebase-ddd-course-1-domain-driven-design-principles/