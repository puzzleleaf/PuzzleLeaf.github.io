---
title: "[Unreal] 5.x 버전에서 StaticLoadClass가 동작하지 않는 경우"
excerpt: "Unreal Engine 5.x 버전으로 업데이트 하고 Blueprint를 c++ 소스 코드로 읽어오는 방식이 동작하지 않는 이슈가 발생했다."
header:
  overlay_image: https://images.unsplash.com/photo-1498050108023-c5249f4df085?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1000&q=80
  overlay_filter: 0.5
  caption: "Photo credit: [**Unsplash**](https://images.unsplash.com/photo-1498050108023-c5249f4df085?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1000&q=80)"
categories: Unreal
tags: Unreal
toc: true
toc_label: "목차"
toc_sticky: true 
toc_icon: "list"
---

# 개요
Unreal Engine 5.x 버전으로 업데이트 하고 Blueprint를 c++ 소스 코드로 읽어오는 방식이 동작하지 않는 이슈가 발생했다.

# 이슈
플러그인의 Content 폴더안에 있는 blueprint 파일을 아래와 같은 c++ 소스로 읽어오는데 에디터 환경에서는 정상 동작하고 Windows 빌드로 뽑았을때 리소스를 찾지 못하는 현상이 발생했다.

<deckgo-highlight-code language="cpp" line-numbers="" terminal="carbon" theme="dracula" editable="false">
<code slot="code">const FString FProtectionUI::BackgroundBlueprintPath = TEXT("/MyPlugin/Blueprint/BackgroundBlueprint.BackgroundBlueprint_C");

UClass* backgroundWidgetClass = StaticLoadClass(UObject::StaticClass(), nullptr, *BackgroundBlueprintPath);
</code>
</deckgo-highlight-code>

# 해결
Unreal Engine 5.x 버전부터 뭔가 바뀐거 같아서 찾아봤다.

프로젝트 세팅 -> 패키징 -> 쿠킹할 추가 에셋 디렉터리 목록에서 해당하는 플러그인의 Content를 포함해줘야 실제 빌드에서도 정상적으로 동작한다.

![](https://velog.velcdn.com/images/puzzleleaf/post/6980fc94-c636-4455-ad2f-3f26a18a939d/image.png)

