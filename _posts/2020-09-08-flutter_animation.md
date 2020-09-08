---
title: "Flutter(플러터) Animation 문서 읽기"
categories: flutter
tags: [flutter, animation, document]
toc: true  
toc_sticky: true 
---

# 개요
https://api.flutter.dev/flutter/widgets/TweenAnimationBuilder-class.html

# TweenAnimationBuilder <T> 클래스
대상 값이 변경 될 때마다 위젯 속성을 대상 값으로 애니메이션 하는 위젯 빌더입니다.    
애니메이션의 속성 유형은 제공된 트윈 유형을 통해 정의됩니다.


## Tween 유형
**ColorTween**    
두 가지 색상 사이의 보간.    
이 유형은 Color.lerp를 사용하기 위한 보간법을 전문으로 한다.
~~~dart
ColorTween({ Color begin, Color end }) : super(begin: begin, end: end);
~~~
[Code Pen](https://codepen.io/puzzleleaf/pen/YzqexJR)
{% gist PuzzleLeaf/9d8de01ac237e08f432a59c744636e92 %}    

