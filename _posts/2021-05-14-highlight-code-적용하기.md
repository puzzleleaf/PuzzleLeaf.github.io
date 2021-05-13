---
title: "블로그에 highlight code 적용하기"
categories: Blog
tags: Blog
toc: true  
toc_sticky: true 
---

---
# 개요

블로그를 처음부터 깔끔하게 관리하고 싶어서 새로 만들게 되었습니다.    
아무래도 개발 중심의 블로그가 될 것 같아서 highlight code 관련해서 먼저 설정을 해봤습니다.    

## Heighlight Code

### 찾아보기
이것저것 찾아보다가 웹 오픈소스 에디터인 [DeckDeckGo의 문서](https://docs.deckdeckgo.com/?path=/docs/components-highlight-code--highlight-code) 에서 highlight code 관련 예제를 찾게 되었습니다.


아래 이미지를 보면 뭔가 조금 더 개발자스러운(?) 느낌으로 적용할 수 있을 것 같아서 이것을 블로그에 적용해보기로 했습니다.

<p style="text-align: center">
  <img src="https://user-images.githubusercontent.com/20354164/118154391-f32b8b80-b451-11eb-8060-d55b0e359cc2.png" alt>
  <span>Carbon 스타일</span>
</p>

<p style="text-align: center">
  <img src="https://user-images.githubusercontent.com/20354164/118154889-92508300-b452-11eb-94a2-e687297b6ac4.png" alt>
  <span>Ubuntu 스타일</span>
</p>

### 적용하기
DeckDeckGo 공식 문서에서는 다양한 방법을 제공하고 있는데, 그중 CDN을 통해서 제공하는 스크립트를 사용하는 방식으로 블로그에 적용해보려고 합니다.

[공식 문서](https://docs.deckdeckgo.com/?path=/docs/components-highlight-code--highlight-code)의 아래의 메뉴에 script 주소를 확인할 수 있습니다. 스크립트를 블로그에 추가하기만 하면 설정은 끝 입니다.

~~~ html
<script type="module" src="https://unpkg.com/@deckdeckgo/highlight-code@latest/dist/deckdeckgo-highlight-code/deckdeckgo-highlight-code.esm.js"></script>
~~~

실제 사용시 아래와 같이 deckgo-highlight-code에 언어와 테마 등 설정들을 적용할 수 있습니다. 그리고 code부분에 원하는 코드를 추가할 수 있습니다.

~~~ html
<deckgo-highlight-code language="javascript" line-numbers="" terminal="carbon" theme="dracula" editable="false">
<code slot="code">Your Code....
</code>
</deckgo-highlight-code>
~~~

<deckgo-highlight-code language="javascript" line-numbers="" terminal="carbon" theme="dracula" editable="false">
<code slot="code">Your Code....
</code>
</deckgo-highlight-code>

### 스타일링
[예제 사이트](https://docs.deckdeckgo.com/?path=/story/components-highlight-code--highlight-code&args=language:dart;highlightLines)에서 다양한 스타일링을 적용해 볼 수 있습니다.

![image](https://user-images.githubusercontent.com/20354164/118158009-3556cc00-b456-11eb-80cc-202abf03abe1.png)