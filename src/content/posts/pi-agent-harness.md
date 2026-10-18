---
title: "Pi 하네스 분석: 내 Claude·Codex 작업에 적용하기"
description: "Codex가 검색 태그 필터를 구현하고 Claude Code가 리뷰하는 사례로 Pi를 읽습니다. 모델 연결과 CLI 연결을 구분하고, Pi가 맡는 실행과 직접 구현할 연결 코드를 소스·흐름도로 따라갑니다."
summary:
  - "Pi는 모델 요청, 도구 실행, 결과 전달을 이어 주는 하네스입니다. Claude Code·Codex도 각자의 하네스를 가진 도구입니다."
  - "Pi에서 Claude·OpenAI 계열 모델을 호출하는 구성과, 기존 Claude Code·Codex CLI를 도구로 부르는 구성은 다릅니다."
  - "검색 태그 필터의 구현 → 리뷰 → 보완 → 재검토를 실제 Pi 함수에 연결해 따라갑니다."
  - "다른 도구의 대화가 자동으로 전달되지는 않습니다. 목표·변경 내용·검토 기준을 작업 명세로 만들어 넘겨야 합니다."
  - "Pi는 도구 호출의 공통 실행을 맡고, CLI 시작·출력 해석·권한·중단·세션 연결은 프로젝트에서 구현합니다."
  - "Claude의 리뷰와 프로젝트 검사는 함께 실행할 수 있습니다. 빌드 통과와 기능 완성은 구분해야 합니다."
  - "애니메이션의 현재 담당과 실제 함수·파일·줄 번호를 함께 강조하고, 직접 구현할 함수와 Pi 연결점을 구분합니다."
  - "직접 연결과 Pi 사용을 같은 사례로 비교합니다. 고정 순서는 작은 스크립트로, 모델이 다음 일을 고르는 흐름은 Pi의 공통 실행 루프로 구성할 수 있습니다."
date: 2026-10-17
updated: 2026-10-18
category: 소스 코드 분석
tags: [AI, Pi, Claude, Codex, 에이전트, 하네스, TypeScript, 소스 코드 분석]
featured: true
draft: false
interactive: [pi-harness]
---

## Pi는 무엇인가

블로그 검색에 **태그 필터**를 추가한다고 해 봅시다. Codex가 기능을 구현하고, Claude Code가 “검색어를 바꾸면 선택한 태그가 풀립니다”라고 지적합니다. 이 내용을 Codex에게 돌려보내 보완하고, 새 수정안을 다시 검토하려면 두 도구 사이에서 작업과 결과를 이어 줄 코드가 필요하겠죠.

[Pi](https://github.com/earendil-works/pi)는 이런 연결을 만들 때 사용할 수 있는 에이전트 도구 모음입니다. 모델에게 요청을 보내고, 모델이 선택한 도구를 실행하고, 결과를 다음 판단으로 이어 줍니다. 터미널에서 코딩 에이전트로 사용하거나 내 프로그램에서 함수로 불러 쓸 수 있습니다. [공식 개요](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/README.md)

모델 주변에서 실행을 관리하는 이 코드를 <strong>하네스(harness)</strong>라고 부릅니다. 모델이 다음 할 일을 판단하면, 하네스가 도구와 기록을 챙기는 식입니다. 먼저 아래 그림으로 역할을 살펴보겠습니다.

<figure class="pi-note" aria-label="나의 요청, Pi 하네스, AI 모델과 도구의 관계">
  <figcaption><span class="note-index">그림 01 · 등장인물부터 만나기</span><strong>생각하는 모델 옆에, 일을 잇는 Pi.</strong></figcaption>
  <div class="note-cast">
    <div class="note-person">
      <svg class="note-doodle" viewBox="0 0 90 80" aria-hidden="true"><path class="doodle-peach" d="M17 13q27-7 52 1l3 38-28 4-15 12 1-14-14-3z"/><path class="doodle-stroke" d="M17 14q26-5 51 0l2 38-28 3-14 12 1-14-13-2zM27 28h29M27 38h20"/><circle cx="57" cy="58" r="10" class="doodle-fill"/><path class="doodle-stroke" d="M48 59q9 5 18-1M73 12l4-5M77 23h5"/></svg>
      <strong>나의 요청</strong><small>“파일을 읽어 줘”</small>
    </div>
    <span class="note-arrow" aria-hidden="true">→</span>
    <div class="note-person note-person--pi">
      <svg class="note-doodle" viewBox="0 0 90 80" aria-hidden="true"><path class="doodle-fill" d="M21 13l43-2 5 55-44 3z"/><path class="doodle-stroke" d="M21 13l43-2 5 55-44 3zM28 14l3 54M16 23h11M17 35h11M18 48h11M19 61h11M40 23l15-1M45 49q6 5 11-1"/><circle cx="42" cy="38" r="1.6" fill="currentColor"/><circle cx="55" cy="37" r="1.6" fill="currentColor"/><path class="doodle-peach" d="M59 6l15 2-2 17-15-2z"/><path class="doodle-stroke" d="M73 57l7 6M77 53l6 3"/></svg>
      <strong>Pi 하네스</strong><small>자료와 실행을 챙겨요</small>
    </div>
    <span class="note-arrow" aria-hidden="true">⇄</span>
    <div class="note-person">
      <svg class="note-doodle" viewBox="0 0 90 80" aria-hidden="true"><path class="doodle-lilac" d="M22 21q8-18 24-9 18-7 23 12 14 3 9 19-2 13-19 13H28Q9 58 11 42q-4-16 11-21z"/><path class="doodle-stroke" d="M22 21q8-18 24-9 18-7 23 12 14 3 9 19-2 13-19 13H28Q9 58 11 42q-4-16 11-21zM43 43q5 5 10 0M28 65l26 1M35 70h13M72 8v-6M69 5h6"/><circle cx="34" cy="34" r="1.7" fill="currentColor"/><circle cx="57" cy="34" r="1.7" fill="currentColor"/></svg>
      <strong>AI 모델</strong><small>다음 할 일을 판단해요</small>
    </div>
  </div>
  <div class="note-tool-link" aria-hidden="true"></div>
  <div class="note-toolbox"><svg class="note-doodle" viewBox="0 0 44 42" aria-hidden="true"><path class="doodle-stroke" d="M5 15h33l1 22H4zM14 15V8h15v7M4 23h35M18 21v6h7v-6"/></svg><div><strong>Pi가 여는 도구 상자</strong><small>파일 읽기 · 검색 · 필요한 명령 실행</small></div></div>
  <p class="note-caption">모델이 “읽어 줘”라고 요청하면, Pi가 도구를 실행해 결과를 돌려줍니다. 역할을 이해하기 위해 단순화한 그림입니다.</p>
</figure>

이 글의 질문은 <strong>“이미 Claude와 Codex를 쓰는 내 개발 작업에 Pi를 어떻게 적용할까?”</strong>입니다. 작은 기능 개발 사례에서 시작해 실제 소스를 읽고, Pi에서 가져다 쓸 부분과 직접 만들 부분을 나누겠습니다. 중간의 작업 장면은 화면에 들어오면 반복 재생됩니다. 멈춰서 읽거나 작업 조건을 바꿔 흐름을 살펴볼 수 있습니다.

처음에는 **모델은 다음 일을 고르고, Pi는 그 일을 도구에 연결하고, 내 코드는 실제 실행 방법을 만든다**는 흐름만 기억해 주세요. 함수 이름은 외우지 않아도 됩니다. 장면을 보다가 궁금한 단계에서 소스 링크를 열어 확인하면 됩니다.

여기서 **도구**는 하네스에 등록한 실행 함수이고, **문맥**은 모델에게 함께 보내는 자료입니다. “수정안을 검토해 줘”라는 요청과 검토 기준을 문맥으로 보내면, 모델이 검토 도구를 골라 실행을 요청하는 식입니다.

## 먼저 Claude·Codex와 연결한다는 뜻부터 나눠 봅시다

Claude라는 모델을 호출하는 것과 Claude Code라는 코딩 도구를 실행하는 것은 다릅니다. Codex도 모델 공급자 연결과 코딩 도구 연결을 구별해서 보아야 합니다. 이번 글에서 **기존에 쓰는 도구를 이어 붙이는 사례는 Claude Code와 Codex CLI를 기준**으로 삼겠습니다. 채팅 앱이나 데스크톱 앱의 현재 대화를 Pi가 그대로 조종하는 구성은 아닙니다.

CLI는 터미널 명령으로 실행하는 프로그램을 뜻합니다. 이 사례에서는 내 연결 코드가 Codex와 Claude Code를 명령으로 실행하고 결과를 받습니다.

| 원하는 적용 | 실제 구조 | 실행을 맡는 하네스 |
| --- | --- | --- |
| 같은 도구·문맥 정책으로 모델을 바꾸기 | 내 앱 → Pi → Claude 또는 지원되는 OpenAI 계열 모델 | Pi가 모델 요청과 도구 루프를 관리합니다. |
| Codex의 구현과 Claude Code의 리뷰 이어 붙이기 | 내 앱 → Pi의 도구 호출 → 직접 만든 연결 코드 → 해당 CLI | 바깥 흐름은 Pi, CLI 안의 작업은 해당 도구가 관리합니다. |
| 지금 쓰는 도구의 작업 품질부터 개선하기 | 작업 명세·완료 기준을 Claude Code·Codex에 전달 | 기존 도구를 그대로 사용할 수 있습니다. Pi 도입은 선택입니다. |

Pi의 [`anthropic` 공급자](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/ai/src/providers/anthropic.ts#L74-L90)와 [`openai-codex` 공급자](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/ai/src/providers/openai-codex.ts#L7-L22)는 첫 번째 구조의 모델 연결입니다. `openai-codex`라는 이름이 Codex 앱을 실행한다는 뜻은 아닙니다. 사용할 모델·인증 방식은 [공급자 문서](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/docs/providers.md)와 실제 사용 환경에서 확인해야 합니다.

아래의 중심 사례는 두 번째 구조입니다. **Codex가 기능을 구현하고 Claude Code가 리뷰한 뒤, 필요한 보완을 Codex에 돌려보내는 흐름**을 보겠습니다. Pi는 바깥의 모델·도구 루프를 맡습니다. 어느 도구가 작성이나 리뷰에 더 뛰어나다는 비교 결과를 전제하지 않습니다.

## 어떤 기준으로 분석했나

Pi의 소스는 **2026년 10월 17일에 확인한 커밋** [`8efd314`](https://github.com/earendil-works/pi/commit/8efd314fa3d3232ce197d8d7a6a5660530b9d83e)를 기준으로 고정했습니다. Claude Code·Codex CLI 연결 방식은 **2026년 10월 18일에 확인한 공식 문서**를 사용했습니다. Pi 구현에서 확인한 사실과 이 글에서 제안하는 연결 구조를 구분해 읽어 주세요.

| 기준 | 확인할 질문 | 읽은 부분 |
| --- | --- | --- |
| 책임 분리 | 모델 통신·도구 실행·세션은 어디서 나뉘나요? | `sdk.ts`, `agent.ts`, `agent-loop.ts` |
| 문맥 구성 | 기록 중 무엇을 이번 모델 요청에 보내나요? | `session-manager.ts`, `messages.ts`, `runner.ts` |
| 도구 계약 | 입력 검증·실행·오류를 어떻게 연결하나요? | `agent-loop.ts`, 도구 정의 타입 |
| 연결 방식 | 같은 프로세스의 SDK와 별도 프로세스의 RPC는 어떻게 다른가요? | SDK·CLI 통합 문서 |
| 기존 도구 활용 | Claude Code·Codex를 호출하려면 무엇을 직접 만드나요? | 각 CLI 공식 문서와 Pi의 `execute()` 호출 지점 |
| 적당한 구성 | 무엇부터 만들고 어떤 기능은 미뤄도 될까요? | README의 방향과 실행·확장 경계 |

방법은 **공식 문서와 구현을 대조하는 소스 분석**입니다. 실제 모델·CLI를 연결한 통합 실험이나 비용·속도 비교는 수행하지 않았습니다. 예제 코드는 연결 지점을 설명하는 응용 예제이고, 장면의 응답은 미리 작성한 것입니다.

## 그림에서 구분할 네 가지 역할

| 장면의 표시 | 이 사례에서 맡는 일 | 코드와의 연결 |
| --- | --- | --- |
| Pi가 처리 · 초록색 | 모델을 부르고, 도구 입력을 검사하고, 결과를 다음 요청에 연결합니다. | [`runLoop()`](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L170-L299) 같은 Pi 구현으로 연결합니다. |
| 내가 구현 · 살구색 | 작업 명세를 만들고, CLI를 실행해 결과를 읽고, 화면에 표시합니다. | 내 함수는 직접 작성합니다. 소스 링크는 Pi가 내 함수를 부르는 [`tool.execute()` 연결점](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L821-L857)을 보여 줍니다. |
| 모델이 판단 · 보라색 | 구현·리뷰·보완 중 다음 작업을 고르고, 결과를 보고 답변을 정합니다. | 링크는 모델의 사고 과정 대신 Pi가 [응답에서 도구 호출을 받는 곳](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L258-L279)을 가리킵니다. |
| 외부 도구가 처리 · 기본색 | Codex가 코드를 구현하고 Claude Code가 수정안을 리뷰합니다. | 이 내부 구현은 Pi 소스에 없습니다. 내 CLI 연결 코드를 거쳐 실행됩니다. |

그림의 ‘판단 모델’은 Pi가 직접 부르는 모델입니다. 구현·리뷰를 맡긴 Codex·Claude Code는 도구 뒤에 있는 별도 프로그램입니다. 이 둘을 같은 상자로 합치면 누가 기록과 권한을 관리하는지 헷갈리기 쉽습니다.

장면의 숫자는 전체 작업 중 현재 위치입니다. 강조된 번호 아래에서 **현재 담당 → 하는 일 → 함수·파일·줄 번호** 순서로 읽어 보세요. 함수 링크에 초점을 옮기거나 누르면 그 단계에서 재생이 멈추므로, 소스를 읽고 돌아와도 같은 장면을 확인할 수 있습니다.

## 사례 1: Codex가 검색 태그 필터를 만들고 Claude가 리뷰하기

이 블로그의 검색 화면에는 아직 별도 태그 선택기가 없습니다. 다만 `src/pages/search.json.ts`는 이미 글의 태그를 내보내고, `src/layouts/Base.astro`의 검색 코드는 제목·설명·태그·본문을 함께 검색합니다. **기존 검색 데이터는 그대로 쓰고, 검색 화면에 태그 선택과 필터 조건을 더하는 작업**으로 범위를 잡겠습니다.

예를 들어 `AI` 태그를 고른 뒤 검색어를 `Pi`에서 `검색`으로 바꿔도 태그 선택이 유지되어야 합니다. 이 작은 사용 흐름을 완료 기준으로 삼으면, Codex의 구현과 Claude의 리뷰가 무엇을 확인할지 분명해집니다.

| 완료 기준 | 독자가 화면에서 확인할 동작 |
| --- | --- |
| 검색어와 태그를 함께 적용 | `검색`을 입력하고 `AI`를 고르면 두 조건에 맞는 글만 보입니다. |
| 검색어를 바꿔도 선택 태그 유지 | 검색어가 바뀌어도 `AI` 선택은 남아 있습니다. |
| 일치하는 글이 없을 때 안내 | 빈 목록 대신 검색 조건에 맞는 글이 없다는 안내가 보입니다. |

<figure class="pi-note" aria-label="Codex 구현, Claude 리뷰, Codex 보완, 재검토의 네 단계">
  <figcaption><span class="note-index">그림 02 · 기능 하나를 완성하는 여정</span><strong>만들고, 살펴보고, 발견한 문제를 고쳐요.</strong></figcaption>
  <ol class="note-route">
    <li><span class="note-step">01</span><span class="note-who">Codex</span><div class="note-bubble"><strong>“검색 화면에 태그 필터를 만들었어요.”</strong><small>내 연결 코드가 구현 결과 v1과 변경 내용을 보관합니다.</small></div></li>
    <li><span class="note-step">02</span><span class="note-who">Claude</span><div class="note-bubble note-bubble--model"><strong>“검색어를 바꾸면 태그가 풀려요.”</strong><small>같은 v1과 완료 기준을 받아 사용 흐름을 리뷰합니다.</small></div></li>
    <li><span class="note-step">03</span><span class="note-who">Codex</span><div class="note-bubble"><strong>“태그 상태를 유지하도록 보완했어요.”</strong><small>리뷰 내용을 받아 고치고 새 구현 결과 v2를 만듭니다.</small></div></li>
    <li><span class="note-step">04</span><span class="note-who">Pi</span><div class="note-bubble note-bubble--pi"><strong>“새 결과를 다시 리뷰에 연결할게요.”</strong><small>등록한 도구의 결과를 다음 모델 요청으로 전달합니다.</small></div></li>
  </ol>
  <p class="note-caption"><span class="note-stamp">기억할 점</span> Pi가 두 코딩 도구를 기본으로 묶어 주는 것은 아닙니다. 내 도구를 등록하면, 공통 실행 루프를 재사용할 수 있습니다.</p>
</figure>

아래 장면에서는 검색 결과의 변화를 함께 보겠습니다. ‘Claude가 보완점을 찾음’에서는 태그가 풀렸다가 Codex의 보완 뒤 유지됩니다. ‘첫 구현이 리뷰를 통과함’과 ‘Claude 실행이 실패함’을 고르면 그 뒤 작업이 어떻게 달라지는지도 볼 수 있습니다.

<figure data-pi-lab="loop" aria-label="Codex가 검색 태그 필터를 구현하고 Claude Code의 리뷰를 받아 보완하는 작업 장면">
  <figcaption>개발 사례 01 · 검색 태그 필터 개발</figcaption>
  <div data-lab-mount><div class="story-fallback"><p>Codex가 검색 태그 필터 v1을 구현합니다.</p><p>Claude Code는 검색어를 바꿀 때 선택 태그가 풀리는 문제를 발견합니다.</p><p>Pi가 도구 결과를 다음 판단에 연결하면, Codex가 보완하고 Claude Code가 v2를 재검토합니다. 프로젝트 검사 결과까지 확인해 정리합니다.</p></div></div>
</figure>

### 이 장면을 Pi의 코드에 연결하기

`implement_feature`·`review_change`·`run_checks`는 이 글에서 제안하는 도구 이름입니다. 각각 Codex 구현, Claude 리뷰, 프로젝트 검사를 내 연결 코드에 맡깁니다. **Pi는 이 도구들을 검증하고 실행한 뒤, 결과를 다음 모델 판단에 연결하는 공통 경로를 제공합니다.**

예를 들어 장면이 ‘호출 검증’일 때는 초록색으로 `prepareToolCall()`을 강조합니다. ‘CLI 연결’로 넘어가면 살구색으로 바뀌며, 직접 작성할 `runCodex()`·`runClaudeReview()`와 Pi 쪽 호출 경계를 나란히 보여 줍니다. 색이 바뀌는 지점이 구현 책임을 나누어 볼 지점입니다.

장면은 개발 과정을 따라가기 쉽도록 묶어서 보여 줍니다. 첫 구현에서 펼쳐 본 모델 요청·도구 검증·실행·결과 연결은 리뷰와 보완에서도 반복됩니다. 뒤쪽 장면에서 같은 세부 단계를 모두 다시 펼치지는 않았습니다.

| 장면의 동작 | Pi가 제공하는 코드 | 프로젝트에서 구현할 일 |
| --- | --- | --- |
| “검색 태그 필터를 구현하고 리뷰해 줘”라고 요청합니다. | [`AgentSession.prompt()`](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/agent-session.ts#L2058-L2147) → [`Agent.prompt()`](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent.ts#L371-L445) | 작업 ID와 변경 내용, 완료 기준을 고릅니다. |
| 판단 모델이 구현·리뷰·보완을 요청합니다. | [`streamAssistantResponse()`](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L381-L410), [`toolCall` 분기](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L258-L279) | 세 도구의 목적과 선후 조건을 정의합니다. 실행 순서를 강제할 정책은 앱에도 둡니다. |
| 도구 이름과 입력을 확인합니다. | [`prepareToolCall()`](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L708-L780)의 `validateToolArguments()`와 `beforeToolCall` | 현재 작업·버전·실행 횟수에 맞는 호출인지 검사하는 정책을 작성합니다. |
| Codex 구현 또는 Claude 리뷰를 실행합니다. | [`executePreparedToolCall()`](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L821-L857)이 내 `execute()`를 호출합니다. | CLI 시작·작업 디렉터리·권한·출력 해석·시간 제한·중단을 구현합니다. CLI 내부 루프는 해당 도구가 맡습니다. |
| 검토 결과를 다음 판단에 연결합니다. | [`createToolResultMessage()`](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L930-L949)와 [`runLoop()`](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L170-L299) | 결과의 공통 형식과 화면 표시를 정합니다. |

여기서 Pi가 줄여 주는 코드는 **도구 검증 → 실행 이벤트 → 결과 메시지 → 다음 모델 요청**의 반복입니다. 리뷰에서 보완점이 나오면 같은 실행 경로로 Codex를 다시 부를 수 있습니다. 처음부터 문제없이 통과하면 보완 단계를 건너뛸 수 있습니다. 이런 분기를 처리할 때마다 모델 통신과 도구 루프를 다시 작성하지 않아도 됩니다.

반면 **“v1 리뷰가 끝난 뒤에만 보완하기”, “보완은 한 번까지만”, “v2에는 v2의 리뷰·검사를 붙이기”는 내 프로젝트의 규칙**입니다. 버전 식별자도 앱이 만듭니다. Pi가 자동으로 개발 순서나 기능 완성을 보장해 주지는 않습니다. Claude의 JSON과 Codex의 JSONL을 공통 결과 형식으로 바꾸는 일도 연결 코드에 남습니다.

검토에서 문제를 찾은 것은 정상적인 업무 결과입니다. CLI 실행 실패를 ‘문제 없음’으로 처리하면 안 되겠죠. Pi는 도구에서 던진 예외를 오류 결과로 연결할 수 있지만, CLI의 실패를 찾아내는 것은 내 코드의 일입니다. 모델 자체의 `error`·`aborted` 응답은 코어 루프를 끝냅니다. [도구 예외 처리](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L821-L857), [모델 오류의 종료 경로](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L245-L255)

## Pi 없이 직접 연결하면 무엇이 달라질까

방금 본 **검색 태그 필터 구현 → Claude 리뷰 → 필요한 보완 → 재검토**는 Pi 없이도 만들 수 있습니다. 여기서는 두 가지 요구를 나누어 비교하겠습니다. 정해진 순서를 실행하는 것과, 결과를 읽은 모델이 다음 작업을 고르게 하는 것은 필요한 코드가 다르기 때문입니다.

### 순서가 정해져 있다면 작은 연결 스크립트로 충분합니다

내 스크립트에서 `runCodex()`를 호출하고, 그 결과를 `runClaudeReview()`에 넘기면 됩니다. 리뷰 결과를 정한 형식으로 받아 `needsRevision` 같은 값을 검사하면, 보완과 재리뷰도 조건문으로 연결할 수 있습니다.

<figure class="pi-note direct-recipe" aria-label="Pi 없이 고정한 순서로 Codex 구현과 Claude 리뷰를 연결하는 흐름">
  <figcaption><span class="note-index">비교 01 · 순서가 정해진 직접 연결</span><strong>이 순서로 끝난다면, 연결 스크립트 하나.</strong></figcaption>
  <ol class="recipe-route">
    <li><span>내 코드 · 1</span><strong>Codex 구현</strong><code>runCodex(task)</code><small>태그 필터 v1 보관</small></li>
    <li><span>내 코드 · 2</span><strong>Claude 리뷰</strong><code>runClaudeReview(v1)</code><small>태그 유지 문제 발견</small></li>
    <li><span>내 코드 · 조건 분기</span><strong>보완점이 있으면 Codex</strong><code>if (needsRevision)</code><small>리뷰 전달 → v2 보관</small></li>
    <li><span>내 코드 · 마지막</span><strong>재리뷰·검사</strong><code>review(v2) → checks(v2)</code><small>보완이 없으면 v1 검사</small></li>
  </ol>
  <p class="note-caption">함수 이름은 연결 구조를 설명하는 예시입니다. 각 함수의 CLI 실행·출력 해석과 버전 관리는 직접 구현합니다. 바깥에서 다음 행동을 고르는 별도의 판단 모델은 필요하지 않습니다.</p>
</figure>

이 구성은 Codex와 Claude의 내부 하네스를 그대로 사용합니다. **Pi를 쓰지 않는다고 두 도구의 내부 모델·도구 루프까지 새로 만들어야 하는 것은 아닙니다.** 이 정도의 작업에 Pi를 더하면 판단 모델의 호출·인증·문맥 관리가 추가됩니다. 따라서 이 짧은 스크립트보다 항상 코드가 줄어든다고 말할 수는 없습니다.

### 다음 행동을 모델이 고르게 하려면 공통 루프가 필요합니다

요구가 조금 달라졌다고 해 봅시다. Claude가 태그 유지 문제를 찾으면 Codex에게 보완을 맡기고, 리뷰 실행이 실패하면 상태를 설명하게 하고 싶습니다. 이후에는 파일 읽기 도구도 추가해, 근거가 부족할 때 관련 코드를 더 살펴보게 할 수 있겠죠. 이런 **다음 행동의 선택을 바깥 판단 모델에 맡기는 구성**을 양쪽에서 똑같이 만든다면, Pi가 대신하는 부분을 비교할 수 있습니다.

아래에서 살구색은 직접 작성할 코드, 초록색은 Pi에서 가져다 쓰는 코드입니다. 같은 작업이 진행되면 양쪽의 현재 단계가 함께 강조됩니다. 위쪽의 코드 짝에서 직접 만들 함수와 대응하는 Pi 함수·소스 위치를 확인해 보세요. 소스 링크를 읽을 때는 그 단계에서 멈춥니다. 특히 가운데의 CLI 연결 상자를 보세요. 이 부분은 Pi를 써도 남습니다.

<figure class="pi-note dev-flow harness-comparison" data-pi-lab="comparison" aria-label="같은 판단 모델을 사용하는 직접 연결과 Pi 사용에서 구현할 코드를 비교하는 흐름도">
  <figcaption><span class="note-index">비교 02 · 같은 기능, 같은 판단 모델</span><strong>초록색으로 바뀐 공통 실행 코드를 덜 만듭니다.</strong></figcaption>
  <div class="flow-legend"><span class="flow-tag flow-tag--app">내가 구현 · 두 방식 모두 남는 부분 포함</span><span class="flow-tag flow-tag--pi">Pi 제공 · 직접 만들 공통 코드를 대체</span></div>
  <div class="compare-task"><strong>같은 작업을 요청합니다</strong><span>“Codex로 검색 태그 필터를 만들고 Claude로 리뷰해 줘.”</span><small>리뷰에서 태그 유지 문제 발견 → Codex 보완 → v2 재리뷰·검사</small></div>
  <div class="compare-row compare-headings"><div><strong>Pi 없이 · 직접 연결</strong><small>바깥 판단 모델의 실행 루프도 내가 작성</small></div><div><strong>Pi 사용 · 공통 루프 재사용</strong><small>같은 판단 모델에 등록한 도구를 연결</small></div></div>
  <div class="compare-row">
    <div class="flow-node flow-node--app" data-compare-step="request"><span class="flow-owner">직접 연결 · 내가 구현 A</span><strong>문맥을 넣어 판단 모델 요청</strong><small>요구사항·기록·도구 목록을 모델 입력으로 변환하고 응답 받기</small><code>requestNextStep(context, tools)</code></div>
    <div class="flow-node flow-node--pi" data-compare-step="request"><span class="flow-owner">Pi 사용 · 제공 A</span><strong>모델 요청 경로 재사용</strong><code>streamAssistantResponse()</code><small>문맥 변환과 모델 스트림 호출 연결</small><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L381-L410">agent-loop.ts · 381–410</a></div>
  </div>
  <div class="compare-decision" data-compare-step="choose"><span>↓ 두 방식의 판단 모델이 선택</span><strong>처음에는 implement_feature, 구현 뒤에는 review_change</strong><small>어떤 일을 고를지는 모델의 판단입니다. 실행을 허용할지는 내 정책으로 확인합니다.</small></div>
  <div class="compare-row">
    <div class="flow-node flow-node--app" data-compare-step="dispatch"><span class="flow-owner">직접 연결 · 내가 구현 B</span><strong>도구 이름·인자 검사 후 함수 호출</strong><small>요청한 도구 찾기 → 스키마 검증 → 함수 호출·도구 예외 처리</small><code>validateCall() → dispatchTool()</code></div>
    <div class="flow-node flow-node--pi" data-compare-step="dispatch"><span class="flow-owner">Pi 사용 · 제공 B</span><strong>도구 검증·호출 처리 재사용</strong><code>prepareToolCall() → executePreparedToolCall()</code><small>도구가 던진 예외도 오류 결과로 연결합니다.</small><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L708-L780">검증 · agent-loop.ts 708–780</a><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L821-L857">내 execute() 호출 · 821–857</a></div>
  </div>
  <div class="flow-arrow" aria-hidden="true">↓ 각각 내 실행 함수로 연결</div>
  <div class="flow-node flow-node--app compare-shared" data-compare-step="cli"><span class="flow-owner">두 방식 모두 · 내가 구현하는 CLI 연결</span><strong>runCodex() · runClaudeReview() · runChecks()</strong><small>구현·리뷰 명세 전달, 프로세스 시작, JSON·JSONL 해석, 실패 확인, 시간 제한·중단 처리<br>Codex의 코드 수정과 Claude의 리뷰는 각 도구의 내부 하네스가 실행합니다.</small></div>
  <div class="flow-arrow" aria-hidden="true">↓ 결과 또는 실행 실패 반환</div>
  <div class="compare-row">
    <div class="flow-node flow-node--app" data-compare-step="result"><span class="flow-owner">직접 연결 · 내가 구현 C</span><strong>결과를 모델용 메시지로 연결</strong><small>호출 ID에 결과 대응 → 성공·오류 메시지 구성 → 진행 이벤트 발행</small><code>appendToolResult() · emitProgress()</code></div>
    <div class="flow-node flow-node--pi" data-compare-step="result"><span class="flow-owner">Pi 사용 · 제공 C</span><strong>결과 메시지·실행 이벤트 재사용</strong><code>createToolResultMessage()</code><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L930-L949">결과 메시지 · agent-loop.ts 930–949</a><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L621-L646">병렬 도구 완료 이벤트 · 621–646</a></div>
  </div>
  <div class="compare-arrows" aria-hidden="true"><span>↓</span><span>↓</span></div>
  <div class="compare-row">
    <div class="flow-node flow-node--app" data-compare-step="repeat"><span class="flow-owner">직접 연결 · 내가 구현 D</span><strong>결과를 넣어 다음 모델 요청 반복</strong><small>Claude의 지적을 문맥에 추가 → A로 돌아가기 → 종료·모델 오류 분기</small><code>while (...) { request → execute → result }</code></div>
    <div class="flow-node flow-node--pi" data-compare-step="repeat"><span class="flow-owner">Pi 사용 · 제공 D</span><strong>다음 요청·턴 종료 경로 재사용</strong><code>runLoop()</code><small>결과 메시지를 연결하고 다음 요청으로 돌아갑니다.</small><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L258-L299">결과 반영·반복 판단 · agent-loop.ts 258–299</a><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L245-L255">모델 오류·중단 종료 · 245–255</a></div>
  </div>
  <div class="compare-arrows"><span>↶ 내 반복 코드로 A에 복귀</span><span>↶ Pi의 루프로 A에 복귀</span></div>
  <div class="flow-node flow-node--app compare-shared" data-compare-step="policy"><span class="flow-owner">두 방식 모두 · 내가 정하는 업무 규칙</span><strong>작업 명세·공간 · v1/v2 대응 · 보완 한도 · 완료 기준</strong><small>이 규칙은 실행 전후에 적용합니다. 최신 버전의 리뷰·검사 통과를 확인하고 결과를 표시합니다. 보완 한도나 실행 실패 때문에 멈출 때도 상태를 남깁니다.</small></div>
  <p class="note-caption">직접 연결 쪽 함수 이름은 직접 작성할 역할의 예시입니다. Pi 쪽 링크는 분석 커밋의 실제 구현입니다. 두 열은 같은 판단 모델을 쓰는 구성을 비교하며, 위의 고정 순서 스크립트와는 구분합니다. 일반 경로를 단순화했고, 거절·중단 등 상세 분기는 뒤의 개발자용 흐름도에서 다룹니다.</p>
</figure>

그림의 초록색은 **바깥 하네스에서 직접 구현하던 역할을 Pi로 대체할 수 있는 부분**입니다. 실제로 없애도 되는 파일 수나 코드 줄 수를 측정한 결과는 아닙니다. 다른 SDK로 직접 연결한다면 그 SDK가 제공하는 공통 처리도 비교에 반영해야 합니다.

| 같은 사례에서 필요한 코드 | 직접 연결 | Pi 사용 |
| --- | --- | --- |
| 다음 행동을 고르는 모델 요청·도구 호출 처리 | 직접 작성하거나 다른 SDK를 선택합니다. | 모델·도구를 설정하고 Pi의 요청·실행 루프를 사용합니다. |
| 도구 결과의 메시지 변환·호출 ID 연결·반복 | 직접 연결합니다. | `createToolResultMessage()`·`runLoop()`를 재사용합니다. |
| CLI의 JSON·JSONL 해석·프로세스 정리 | 직접 구현합니다. | 그대로 직접 구현합니다. Pi의 도구 결과 형식에 맞춰 반환합니다. |
| v1/v2 작업 명세·권한·보완 한도·기능 완료 판정 | 직접 구현합니다. | 그대로 직접 구현합니다. |
| 바깥 판단 모델을 위한 설정 | 고정 순서 스크립트라면 필요 없습니다. | 판단 모델·인증·도구 등록·세션 수명을 설정합니다. |

이 사례에서 Pi로 덜 만드는 것은 **여러 도구의 결과를 다음 모델 판단으로 이어 주는 공통 실행 코드**입니다. CLI 연결과 업무 규칙은 계속 직접 만듭니다. “리뷰 후 한 번 보완”이라는 순서가 고정돼 있다면 직접 연결부터 시작하고, 결과에 따라 자료와 도구를 고르는 흐름이 필요할 때 Pi를 검토하겠습니다.

### 같은 사례의 최소 코드를 나란히 읽어 봅시다

위의 두 경로를 코드로 옮기면 아래와 같습니다. 양쪽 모두 같은 목표·세 도구·완료 기준을 사용하고, 바깥 판단 모델이 다음 작업을 고릅니다. <strong>왼쪽의 A·B·C·D를 오른쪽의 도구 등록과 <code>session.prompt()</code>로 대체하는 부분</strong>에 집중해 보세요.

공통 작업 명세와 실행 함수 계약은 아래에 접었습니다. 오른쪽에는 중단 연결·세션 정리를 남기고, Pi 전용 도구 등록·모델 설정은 별도 항목으로 접었습니다. 이 등록 코드도 직접 작성해야 합니다. Pi를 쓰면 실행 루프를 재사용할 수 있지만, 등록 코드까지 없어지는 것은 아닙니다.

<section class="pi-note code-comparison" id="pi-minimal-code" aria-label="같은 검색 태그 필터 사례의 직접 연결과 Pi 사용 최소 코드 비교">
  <div class="code-comparison-grid">
    <section class="code-lane code-lane--direct" aria-labelledby="direct-code-heading">
      <h4 id="direct-code-heading">Pi 없이 · 실행 루프 직접 연결</h4>
      <p class="code-lane-note">A 요청 · B 검증 · C 결과 · D 반복을 직접 작성</p>
<pre data-code-example="direct"><code>async function direct(planner: Planner, signal: AbortSignal) {&#10;  const history: Entry[] = [{ role: &quot;user&quot;, text: brief }];&#10;  // D · 다음 요청으로 돌아오는 반복문도 직접 작성&#10;<span class="code-emphasis" data-code-role="repeat">  for (;;) {</span>&#10;    signal.throwIfAborted();&#10;    // A · 모델 요청과 응답 처리&#10;<span class="code-emphasis" data-code-role="request">    const reply = await planner.next(history, specs, signal);</span>&#10;    if (reply.error) throw new Error(reply.error);&#10;    history.push(reply.message);&#10;    if (!reply.message.calls.length) return reply.message.text;&#10;&#10;    for (const call of reply.message.calls) {&#10;      let result: Outcome;&#10;      try {&#10;        // B · 도구 이름·빈 인자 검사와 실행 분기&#10;        const spec = specs.find(t =&gt; t.name === call.name);&#10;<span class="code-emphasis" data-code-role="dispatch">        if (!spec || JSON.stringify(call.args) !== &quot;{}&quot;) {</span>&#10;          throw new Error(&quot;허용되지 않은 도구 호출&quot;);&#10;        }&#10;        result = await runStep(spec.name, signal);&#10;      } catch (error) {&#10;        signal.throwIfAborted();&#10;        result = { status: &quot;failed&quot;, report: String(error) };&#10;      }&#10;      // C · 호출 ID에 맞춰 모델용 결과 메시지 연결&#10;<span class="code-emphasis" data-code-role="result">      history.push({</span>&#10;        role: &quot;tool&quot;, callId: call.id,&#10;        text: result.report,&#10;        isError: result.status === &quot;failed&quot;,&#10;      });&#10;    }&#10;  }&#10;}</code></pre>
      <details class="code-contract"><summary>직접 연결 쪽의 모델 어댑터 계약</summary><p><code>Planner</code>는 이 글에서 정한 앱의 인터페이스입니다. 공급자 SDK의 요청·응답을 아래 형식으로 바꾸고, 중단 신호를 전달하는 어댑터를 직접 구현해야 합니다. 세 도구의 인자는 빈 객체로 선언합니다. 이 어댑터는 Pi가 대신하는 모델 연결 코드이므로 공통 구현으로 세지 않았습니다.</p>
<pre data-code-example="adapter"><code>type Call = { id: string; name: string; args: unknown };&#10;type Assistant = { role: &quot;assistant&quot;; text: string; calls: Call[] };&#10;type Entry = Assistant&#10;  | { role: &quot;user&quot;; text: string }&#10;  | { role: &quot;tool&quot;; callId: string; text: string; isError: boolean };&#10;interface Planner {&#10;  next(&#10;    history: Entry[], tools: typeof specs, signal: AbortSignal&#10;  ): Promise&lt;{ message: Assistant; error?: string }&gt;;&#10;}</code></pre>
      </details>
    </section>
    <section class="code-lane code-lane--pi" aria-labelledby="pi-code-heading">
      <h4 id="pi-code-heading">Pi 사용 · 도구 등록 후 루프 재사용</h4>
      <p class="code-lane-note">초록색 줄에서 Pi의 A·B·C·D 실행 경로로 진입</p>
<pre data-code-example="pi"><code>async function withPi(model: Model&lt;any&gt;, signal: AbortSignal) {&#10;  const { session } = await createSessionForTask(model);&#10;  const stop = () =&gt; { void session.abort(); };&#10;  signal.addEventListener(&quot;abort&quot;, stop, { once: true });&#10;  try {&#10;    signal.throwIfAborted();&#10;    // A·B·C·D · 요청·검증·결과·반복은 Pi의 루프&#10;<span class="code-emphasis" data-code-role="reused">    await session.prompt(brief);</span>&#10;    signal.throwIfAborted();&#10;    return session.getLastAssistantText();&#10;  } finally {&#10;    signal.removeEventListener(&quot;abort&quot;, stop);&#10;    session.dispose();&#10;  }&#10;}</code></pre>
      <details class="code-pi-setup"><summary>Pi 전용 코드 펼치기 · 도구 등록·모델 설정</summary><p><code>createSessionForTask()</code>는 아래처럼 직접 작성할 앱 함수입니다. 같은 세 실행 함수를 Pi 도구로 등록합니다. 이 코드는 Pi 사용에 추가로 필요하며, 두 방식의 공통 코드와 구분해 접었습니다.</p>
<pre data-code-example="pi-setup"><code>import { Type } from &quot;typebox&quot;;&#10;import {&#10;  createAgentSession, defineTool, SessionManager,&#10;} from &quot;@earendil-works/pi-coding-agent&quot;;&#10;import type { Model } from &quot;@earendil-works/pi-ai&quot;;&#10;&#10;async function createSessionForTask(model: Model&lt;any&gt;) {&#10;  // 내 코드 · 같은 실행 함수를 Pi의 도구로 등록&#10;  const customTools = specs.map(spec =&gt; defineTool({&#10;    ...spec,&#10;    parameters: Type.Object({}, { additionalProperties: false }),&#10;    executionMode: &quot;sequential&quot;,&#10;    async execute(_id, _args, toolSignal) {&#10;      const result = await runStep(spec.name, toolSignal);&#10;      return {&#10;        content: [{ type: &quot;text&quot;, text: result.report }],&#10;        isError: result.status === &quot;failed&quot;,&#10;      };&#10;    },&#10;  }));&#10;  return createAgentSession({&#10;    model, customTools,&#10;    tools: specs.map(t =&gt; t.name),&#10;    sessionManager: SessionManager.inMemory(),&#10;  });&#10;}</code></pre>
      </details>
      <div class="code-source-links"><strong>이 호출 뒤에서 Pi가 맡는 구현</strong><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L381-L410">A · streamAssistantResponse() · 381–410</a><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L708-L780">B · prepareToolCall() · 708–780</a><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L821-L857">B · executePreparedToolCall() · 821–857</a><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L930-L949">C · createToolResultMessage() · 930–949</a><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L258-L299">D · runLoop()의 결과 반영·반복 · 258–299</a></div>
    </section>
  </div>
  <details class="code-common"><summary>공통 코드 펼치기 · 작업 명세·도구 목록·CLI 실행 계약</summary><p>두 버전은 아래 명세와 <code>runStep()</code>을 함께 사용합니다. <code>runStep()</code>은 Codex 구현·Claude 리뷰·프로젝트 검사를 실행하고 결과를 공통 형식으로 반환하는 앱의 함수입니다.</p>
<pre data-code-example="common"><code>const task = {&#10;  id: &quot;search-tag-filter&quot;,&#10;  goal: &quot;검색 태그 필터를 Codex로 구현하고 Claude로 리뷰&quot;,&#10;  acceptance: [&quot;검색어와 태그 함께 적용&quot;, &quot;태그 선택 유지&quot;, &quot;빈 결과 안내&quot;],&#10;};&#10;const brief = `${task.goal}\n완료 기준: ${task.acceptance.join(&quot; · &quot;)}&#10;지적이 있으면 Codex가 한 번 보완하고, 새 버전을 재리뷰·검사해 줘.`;&#10;const specs = [&#10;  { name: &quot;implement_feature&quot;, label: &quot;Codex 구현&quot;,&#10;    description: &quot;기능을 구현하거나 현재 리뷰를 반영해 보완합니다.&quot; },&#10;  { name: &quot;review_change&quot;, label: &quot;Claude 리뷰&quot;,&#10;    description: &quot;현재 구현 버전을 완료 기준에 맞춰 리뷰합니다.&quot; },&#10;  { name: &quot;run_checks&quot;, label: &quot;프로젝트 검사&quot;,&#10;    description: &quot;현재 버전의 빌드·검색 조건을 검사합니다.&quot; },&#10;] as const;&#10;type Step = typeof specs[number][&quot;name&quot;];&#10;type Outcome = { status: &quot;completed&quot; | &quot;failed&quot;; report: string };&#10;&#10;// 앱에서 구현할 계약입니다. CLI 구현은 여기에 포함하지 않았습니다.&#10;declare function runStep(&#10;  step: Step, signal?: AbortSignal&#10;): Promise&lt;Outcome&gt;;</code></pre>
    <p>실제 구현에는 <code>runCodex()</code>·<code>runClaudeReview()</code>·<code>runChecks()</code> 연결, CLI 출력 해석, 작업 공간과 v1/v2 대응, 한 번의 보완 한도, 시간·사용량 한도와 중단 뒤 프로세스 정리가 필요합니다. 이 부분은 두 방식 모두 직접 만듭니다.</p>
  </details>
</section>

직접 연결 쪽은 설명을 위한 앱 인터페이스를, Pi 쪽은 분석 커밋의 SDK API를 사용한 <strong>응용 예제</strong>입니다. 판단 모델의 인증과 공급자 설정은 준비되어 있다고 가정합니다. 정상적인 구현·리뷰·보완 경로를 비교하며, 스트리밍 화면·재시도·영속 저장의 세부 구현은 생략했습니다. 실제 모델·CLI를 실행해 두 코드의 결과가 같음을 검증한 예제는 아닙니다.

같은 <code>runStep()</code> 호출은 양쪽에 남아 있습니다. 차이는 그 호출을 고르고, 검증하고, 결과를 다음 판단에 이어 주는 바깥 코드에 있습니다. 앞의 고정 순서 스크립트에는 이런 모델 루프 자체가 필요하지 않다는 점도 함께 기억해 주세요.

## 사례 2: 다른 도구에 넘길 작업 명세 만들기

Claude Code에 “방금 Codex와 얘기한 대로 검토해 줘”라고만 말하면 충분할까요? 서로 다른 대화에는 같은 정보가 들어 있다고 가정할 수 없습니다. 저는 아래처럼 **목표·변경 내용·검토 기준**을 묶어 전달하겠습니다.

```text
작업: search-tag-filter / 구현 결과: search-tag-v1
목표: 블로그 검색에 태그 필터 추가
변경 파일: src/layouts/Base.astro
참고 파일: src/pages/search.json.ts — 기존 tags 데이터 재사용
담당: Codex 구현 / Claude Code 리뷰
완료 기준:
  - 검색어와 선택 태그가 함께 적용됨
  - 검색어를 바꿔도 선택한 태그가 유지됨
  - 결과가 없을 때 안내가 보임
리뷰 대상: v1으로 고정한 변경 내용과 소스
리뷰 범위: 파일 수정 없이 문제 위치 / 이유 / 확인할 동작 반환
보완: 리뷰를 Codex에 전달, 보완한 v2는 새로 리뷰·검사
```

이것은 이 사례의 작업 명세 예시입니다. 권한을 실제로 제한하는 설정은 실행 코드와 각 도구의 환경에서 따로 적용해야 합니다. 명세에 ‘수정하지 말라’고 적는 것만으로 파일 쓰기가 기술적으로 차단되지는 않습니다.

<figure class="pi-note" aria-label="전체 기록에서 현재 요청에 쓸 문맥을 구성하는 개념도">
  <figcaption><span class="note-index">그림 03 · 보관함에서 책상으로</span><strong>쌓아 둔 기록과, 지금 읽을 자료는 달라요.</strong></figcaption>
  <div class="note-context">
    <div class="note-stack"><strong>기록 보관함</strong><span>지난 대화</span><span>다른 가지의 대화</span><span>도구를 쓴 기록</span><span>설정과 요약</span></div>
    <div class="note-sieve">현재 가지 선택<br>문맥 구성·변환<b aria-hidden="true">→</b></div>
    <div class="note-stack note-stack--input"><strong>이번 요청의 문맥</strong><span>현재 지침</span><span>필요한 대화와 요약</span><span>연결된 도구 결과</span></div>
  </div>
  <div class="note-margin"><mark>화면에서 숨긴 메시지도 모델에게는 전달될 수 있어요.</mark></div>
  <p class="note-caption">보관한 기록에서 현재 대화의 가지를 고르고, 요청에 맞게 구성·변환합니다. 위 항목은 역할을 설명하는 개념 예시입니다.</p>
</figure>

<figure data-pi-lab="context" aria-label="Codex 구현 결과와 완료 기준을 Claude Code에 전달하는 장면">
  <figcaption>개발 사례 02 · 작업 인계</figcaption>
  <div data-lab-mount><div class="story-fallback"><p>보관한 작업 명세: 검색 태그 필터 v1 + 완료 기준</p><p>완료 기준을 함께 전달하면 Claude가 검색어 변경 뒤 태그 유지 여부를 확인합니다.</p><p>기준을 빼면 변경 내용만 받습니다. 화면에서 명세를 접는 것과 전달에서 제외하는 것은 다릅니다.</p></div></div>
</figure>

### Pi의 문맥과 외부 CLI의 입력은 두 번 구분합니다

Pi 안에서는 `SessionManager`가 기록을 보관하고 현재 가지를 골라 요청 문맥을 만듭니다. 요청 직전의 가공 지점도 따로 있습니다. 그러나 **Pi의 기록이 외부 Claude Code·Codex 세션으로 자동 복사되지는 않습니다.** 검토 도구의 `execute()`에서 CLI에 전달할 명세를 별도로 구성해야 합니다.

| 연결할 내용 | 실제 코드 또는 구현 책임 |
| --- | --- |
| Pi 기록에서 현재 대화 선택 | [`buildSessionProjection()`](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/session-manager.ts#L1493-L1499), [현재 가지를 따라가는 코드](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/session-manager.ts#L389-L414) |
| Pi의 다음 모델 요청에 넣을 자료 가공 | SDK의 [`transformContext` 연결](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/sdk.ts#L436-L440) → [`emitContext()`](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/extensions/runner.ts#L1305-L1336) |
| 모델용 메시지 형식으로 변환 | [`convertToLlm()`](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/messages.ts#L148-L173)의 `custom` 처리. 화면에서 숨겼다는 이유로 모델 입력에서도 빠지지는 않습니다. |
| 외부 CLI에 넘길 변경 내용·작업 명세 선택 | 프로젝트의 검토 연결 코드. 위의 Pi 문맥 가공과 별도 작업입니다. |
| Pi 세션과 Claude Code 세션·Codex 스레드의 대응 기록 | 프로젝트의 저장 정책. 식별자를 구분해 보관하고, 필요한 경우 해당 CLI의 재개 기능에 연결합니다. |

이 장면은 첫 사례의 ‘CLI 연결’ 안쪽을 펼쳐 본 것입니다. 명세를 고르고 외부 도구에 넘기는 단계는 살구색으로 표시합니다. 이때의 Pi 링크는 내 코드를 부르는 연결점이며, Pi가 명세 작성과 인계를 대신 구현한다는 뜻은 아닙니다. Pi의 세션 압축과 서로 다른 제품 사이의 대화 인계도 별개의 일입니다. 외부 도구의 전체 대화를 자동으로 동기화하는 대신, 처음에는 명시적인 작업 명세로 시작하는 편이 추적하기 쉽겠습니다.

## 사례 3: Claude 리뷰와 프로젝트 검사를 함께 실행하기

Codex가 v1을 만든 뒤에는 Claude의 리뷰와 프로젝트 검사를 함께 실행할 수 있습니다. **두 작업이 같은 버전을 보고 서로의 결과를 기다리지 않을 때** 병렬 실행을 생각해 볼 만합니다. 검사는 분리된 작업 공간에서 실행하거나 소스를 변경하지 않도록 구성하겠습니다.

이 사례에서는 빌드는 통과하지만 Claude가 태그 선택이 풀리는 문제를 찾습니다. 빌드 성공만으로 사용 흐름까지 완성되었다고 볼 수 없겠죠. 먼저 도착하는 결과와 실행 방식을 바꿔 보며 결과가 모이는 과정을 살펴보세요.

<figure data-pi-lab="tools" aria-label="같은 구현 결과의 Claude 리뷰와 프로젝트 검사 결과가 도착하는 장면">
  <figcaption>개발 사례 03 · 리뷰와 검사의 응답</figcaption>
  <div data-lab-mount><div class="story-fallback"><p>같은 구현 결과 v1에 Claude 리뷰와 프로젝트 검사를 요청합니다.</p><p>프로젝트 검사가 먼저 빌드 통과를 알려도 Claude의 리뷰가 남아 있습니다.</p><p>Claude가 태그 유지 문제를 찾으면, 두 결과를 함께 보고 Codex에 보완을 요청합니다.</p></div></div>
</figure>

### 완료 순서와 기록 순서를 나누어 읽기

| 장면의 동작 | Pi의 구현 | 내 코드에서 지킬 조건 |
| --- | --- | --- |
| 리뷰와 검사를 함께 시작합니다. | [`executeToolCalls()`의 실행 모드 선택](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L508-L526)과 [병렬 실행 준비](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L586-L620) | 모델이 실제로 두 호출을 만들었는지 확인합니다. 같은 구현 버전을 고정하고 독립적인 리뷰·검사로 실행합니다. |
| 먼저 끝난 리뷰·검사 소식을 보여 줍니다. | [완료 시 발생하는 도구 이벤트](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L621-L646) | 호출 ID로 화면의 검토 상태를 갱신합니다. |
| 모델에게 두 결과를 연결합니다. | [`Promise.all()` 뒤 호출 순서대로 만드는 결과 메시지](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L646-L663) | 실행 완료 여부와 검토에서 발견한 내용을 구분해 반환합니다. |

Codex의 구현이 끝나기 전에 같은 변경을 Claude가 리뷰하도록 겹치지는 않겠습니다. 선후 관계가 있는 구현·리뷰와 독립적으로 실행할 수 있는 리뷰·검사를 구분해야 합니다. 장면의 완료 순서는 예시이며 실제 소요 시간을 비교한 결과는 아닙니다.

파일을 쓰는 작업은 같은 체크아웃에서 동시에 실행하지 않도록 별도 작업 공간이나 실행 잠금을 설계하겠습니다. Pi의 도구별 `sequential` 설정은 해당 배치의 실행 순서에 관한 것입니다. 다른 세션·다른 프로세스까지 막아 주는 전역 잠금은 아닙니다.

장면에서 순차 실행을 고르면 소스 표시도 [`executeToolCallsSequential()`](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L530-L584) 경로로 바뀝니다. 이 경로는 도구 하나가 끝날 때마다 결과를 기록합니다. 병렬 경로의 `Promise.all()` 설명을 순차 실행에도 그대로 적용하지 않도록 구분했습니다.

## 내 프로그램과 Pi는 어떻게 연결할까

<figure class="pi-note" aria-label="SDK, RPC, Extension의 세 가지 연결 방식 비교">
  <figcaption><span class="note-index">그림 04 · 내 프로그램과 연결하기</span><strong>Pi를 어디에 두고 싶은가요?</strong></figcaption>
  <div class="note-choices">
    <div class="note-choice"><svg class="note-doodle" viewBox="0 0 68 62" aria-hidden="true"><path class="doodle-soft" d="M9 9h47v45H9z"/><path class="doodle-stroke" d="M9 9h47v45H9zM9 19h47M15 14h2M22 14h2M25 33l-5 5 5 5M40 33l5 5-5 5M35 30l-6 16"/></svg><strong>SDK</strong><p>내 TypeScript 앱 안에<br>Pi를 넣고 싶어요.</p><small>함수로 세션을 만들어요.</small></div>
    <div class="note-choice"><svg class="note-doodle" viewBox="0 0 68 62" aria-hidden="true"><path class="doodle-fill" d="M3 15h23v27H3zM42 15h23v27H42z"/><path class="doodle-stroke" d="M3 15h23v27H3zM42 15h23v27H42zM9 48h12M48 48h12M26 25h14l-4-4M42 34H28l4 4M9 23h9M48 23h9"/></svg><strong>RPC</strong><p>따로 실행한 Pi에게<br>일을 부탁하고 싶어요.</p><small>메시지를 주고받아요.</small></div>
    <div class="note-choice"><svg class="note-doodle" viewBox="0 0 68 62" aria-hidden="true"><path class="doodle-fill" d="M12 25h43v29H12z"/><path class="doodle-stroke" d="M12 25h43v29H12zM12 36h43M29 33v7h10v-7M24 25v-8h19v8M34 16V8M34 11q-10-10-14-3 3 8 14 3M34 11q9-11 14-4-3 8-14 4"/></svg><strong>Extension</strong><p>지금 쓰는 Pi에<br>새 도구를 달고 싶어요.</p><small>실행할 기능을 등록해요.</small></div>
  </div>
  <p class="note-caption">SDK와 RPC는 앱에서 Pi를 부르는 방식입니다. Extension은 Pi의 기능을 늘리는 방식이어서 함께 사용할 수도 있습니다.</p>
</figure>

같은 TypeScript 프로그램에서 세션을 만들면 SDK를, 별도 Pi 프로세스를 제어하면 RPC를 검토할 수 있습니다. Extension은 Pi 안에 도구·이벤트 동작을 추가하는 방법입니다. **Pi RPC는 Pi를 제어하는 통신 방식**이고, Claude Code·Codex를 부르는 연결 코드는 그 안의 도구에 별도로 붙습니다. [SDK 공식 문서](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/docs/sdk.md), [Pi CLI 통합 문서](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/docs/cli-integration.md)

여기서부터 구현 계층으로 바꿔 읽는다면, [세션의 요청 준비](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/agent-session.ts#L789-L815)가 기록과 도구를 갖추고, 에이전트 루프가 실행을 연결하고, [`ModelRuntime.streamSimple()`](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/model-runtime.ts#L723-L748)이 모델 공급자로 요청을 전달합니다.

### SDK: 구현·리뷰·검사의 입구를 나눕니다

아래는 Pi API에 맞춰 쓴 **응용 예제**입니다. 세 도구는 같은 `runStep` 계약으로 연결합니다. 실제 프로세스 실행과 작업 상태 관리는 프로젝트에서 구현해 주입합니다. Pi 패키지·TypeBox·판단 모델의 인증과 설정은 준비되어 있다고 가정합니다.

코드는 앞의 [최소 코드 비교에서 Pi 쪽](#pi-minimal-code)을 보시면 됩니다. 도구를 등록하고 `session.prompt(brief)`로 실행하는 부분입니다.

모델은 세 도구 중 다음에 부를 도구를 고릅니다. 작업 목표·완료 기준은 앱이 전달한 `task`에 묶었습니다. `runStep`은 `implement_feature`를 `runCodex()`에, `review_change`를 `runClaudeReview()`에, `run_checks`를 `runChecks()`에 연결합니다. 임의의 실행 파일·작업 디렉터리는 모델 인자로 받지 않습니다. [도구 정의 계약](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/extensions/types.ts#L579-L694), [SDK 도구 옵션](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/sdk.ts#L62-L94)

이 코드에는 아직 상태 관리가 없습니다. 내 실행기는 작업별 현재 버전을 보관하고, 리뷰할 구현 결과가 없는 호출이나 정한 보완 횟수를 넘은 호출을 거절해야 합니다. 보완하면 새 버전을 만들고, 이전 버전의 리뷰·검사 통과를 새 버전의 통과로 옮기지 않습니다. **모델의 도구 선택과 앱의 실행 허용 판단을 함께 두는 구성**입니다.

첫 예제는 `executionMode: "sequential"`로 순차 실행합니다. 사례 3처럼 리뷰·검사를 병렬로 바꿀 때는 두 도구를 `parallel`로 설정하고, 같은 버전을 독립적으로 확인하는지 먼저 보겠습니다. 이 설정이 다른 세션까지 잠그는 것은 아니므로 앱의 작업 상태 확인은 계속 필요합니다.

`completed`는 도구 실행이 끝났다는 뜻입니다. 리뷰가 문제를 찾았는지, 기능의 모든 완료 기준이 충족됐는지는 `report`와 별도의 결과 정책으로 판단해야 합니다.

Pi 쪽 예제는 `prompt()`를 기다리고, 사용이 끝나면 `dispose()`로 세션을 정리합니다. 진행 상태를 표시하려면 `subscribe()`로 이벤트를 구독하고 그 구독도 정리합니다. `SessionManager.inMemory()`는 이 예제의 선택이며 SDK 기본 저장 정책과 다릅니다. 불러오는 확장·설정까지 통제하려면 [전체 구성 예제](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/examples/sdk/12-full-control.ts)와 [세션 수명 관리](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/docs/sdk.md#session-lifecycle)를 함께 보세요.

### CLI 연결 코드에서 맞춰야 할 계약

공식 문서는 Claude Code의 비대화형 `-p` 실행과 Codex의 `exec` 실행을 제공합니다. 아래는 입력·출력 연결의 입구를 보여 주는 명령 형태입니다. 작업 명세를 전달할 때는 문자열로 셸 명령을 조립하기보다, 프로세스 API의 인자 배열과 표준 입력을 사용하도록 구현하겠습니다.

| 맡길 작업 | 공식 실행 입구 | 내 연결 코드가 해석할 것 |
| --- | --- | --- |
| Codex 구현·보완 | `codex exec --json --sandbox workspace-write "<작업 명세>"` | JSONL 진행 이벤트·턴 완료/실패·변경 내용·스레드 식별자 |
| Claude Code 리뷰 | `claude -p "<리뷰 명세>" --output-format json` | JSON 결과·리뷰 지적·실행 오류·세션 식별자 |
| 프로젝트 검사 | 프로젝트가 정한 검사·빌드 스크립트 | 종료 코드·검사 보고서·검사한 구현 버전 |

Claude Code의 JSON 출력 옵션 자체는 권한 제한이 아닙니다. 실제 도구 허용 범위와 설정은 따로 정해야 합니다. 각 CLI의 로그인·권한·확장 설정도 Pi의 설정과 별개입니다. Codex 구현은 분리한 작업 공간에 한정하고, Claude 리뷰에는 고정한 소스와 변경 내용을 읽기 전용 환경으로 제공하겠습니다. 명령 한 줄이 이런 구성을 모두 만들어 주지는 않습니다. 작업 공간·도구 권한·검사 정책을 마련하는 것은 앱의 책임입니다. [Claude Code 비대화형 실행·출력](https://code.claude.com/docs/en/headless), [Claude Code CLI 옵션](https://code.claude.com/docs/en/cli-reference), [Codex 비대화형 실행·JSONL·샌드박스](https://learn.chatgpt.com/docs/non-interactive-mode)

`runStep`은 정상 종료와 유효한 결과를 확인한 뒤 공통 형식으로 바꿉니다. 오류·시간 초과·출력 형식 오류는 실패로 반환합니다. 중단할 때는 Pi가 전달한 `signal`을 실제 CLI 실행과 자식 프로세스 정리까지 연결해야 합니다. 이미 바뀐 파일을 중단 신호가 되돌려 주지는 않습니다.

### Pi의 접수와 완료도 나누어 봅니다

Pi를 RPC로 부를 때 `prompt`에 대한 성공 응답만으로 전체 작업이 끝났다고 보면 안 됩니다. `message_end`는 메시지 완료이고, 자동 진행까지 마친 상태는 `agent_settled`로 확인합니다. 검토 CLI의 완료, Pi의 후속 판단 완료, 앱의 결과 표시를 각각 연결해야 합니다. [Pi RPC 완료 의미](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/docs/cli-integration.md#control-pi-with-rpc)

## 개발자용 흐름도: Pi와 내 코드의 경계

초록색은 **Pi 제공 처리**, 살구색은 **직접 구현할 연결 코드**, 보라색은 응답에 따른 분기입니다. 각 상자에 역할도 적었습니다. Pi의 함수 링크는 분석 커밋의 실제 구현으로 이동합니다. `runCodex`·`runClaudeReview`·`runChecks`와 세 도구 이름은 이 글에서 제안하는 이름이며 Pi 저장소의 내장 기능이 아닙니다.

<figure class="pi-note dev-flow" aria-label="요청 접수부터 도구 실행과 응답까지, Pi 제공 단계와 프로젝트 구현 단계를 구분한 개발자용 흐름도">
  <figcaption><span class="note-index">그림 05 · 개발자의 작업 지도</span><strong>구현·리뷰를 잇는 Pi와, 실제 작업을 실행하는 내 코드.</strong></figcaption>
  <div class="flow-legend"><span class="flow-tag flow-tag--pi">Pi 제공</span><span class="flow-tag flow-tag--app">내 프로젝트 구현</span><span class="flow-tag flow-tag--decision">응답에 따른 분기</span></div>
  <div class="flow-node flow-node--app"><span class="flow-owner">내 프로젝트 · A</span><strong>기능·완료 기준 선택 → 작업 공간·세션 선택</strong><small>예시 <code>handleRequest()</code> · 작업 명세·실행 범위·검토 횟수 한도 설정</small><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/agent-session.ts#L2058-L2147">Pi에 넘기는 입구: AgentSession.prompt()</a></div>
  <div class="flow-arrow" aria-hidden="true">↓</div>
  <div class="flow-node flow-node--pi" id="pi-flow-request"><span class="flow-owner">Pi · B</span><strong>세션에서 문맥 준비 → 모델 요청</strong><small>어떤 변경 내용과 기준을 보낼지는 내 프로젝트의 정책</small><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/agent-session.ts#L789-L815">prepareRequest · agent-session.ts 789–815</a><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L381-L410">streamAssistantResponse() · agent-loop.ts 381–410</a></div>
  <div class="flow-arrow" aria-hidden="true">↓</div>
  <div class="flow-node flow-node--decision"><span class="flow-owner">Pi가 모델 응답을 확인 · C</span><strong>도구 호출이 들어 있나요?</strong><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L245-L279">runLoop()의 오류·toolCall 분기 · 245–279</a></div>
  <div class="flow-branches">
    <div class="flow-track">
      <div class="flow-edge">C → 예 · 도구 경로 ↓</div>
      <div class="flow-node flow-node--pi"><span class="flow-owner">Pi · D</span><strong>도구 찾기 · 인자 형식 검증</strong><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L708-L780">prepareToolCall() · 708–780</a></div>
      <div class="flow-arrow" aria-hidden="true">↓ 검증 통과</div>
      <div class="flow-node flow-node--app"><span class="flow-owner">내 프로젝트 · E</span><strong>허용 범위 확인 → 구현·리뷰·검사 실행</strong><small><code>implement_feature</code> → <code>runCodex()</code><br><code>review_change</code> → <code>runClaudeReview()</code><br><code>run_checks</code> → <code>runChecks()</code><br>각 CLI의 내부 실행·버전·권한·중단은 앱에서 연결</small><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L821-L857">내 execute를 부르는 Pi 코드 · 821–857</a><a href="https://code.claude.com/docs/en/headless">Claude Code 실행 입구 · 공식 문서</a><a href="https://learn.chatgpt.com/docs/non-interactive-mode">Codex 실행 입구 · 공식 문서</a></div>
      <div class="flow-arrow" aria-hidden="true">↓ 성공 · 검증 거절 · 도구 오류</div>
      <div class="flow-node flow-node--pi"><span class="flow-owner">Pi · F</span><strong>도구 결과를 메시지로 연결</strong><small>D에서 거절된 호출은 E를 건너뛰고 오류 결과로 연결됩니다.</small><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L930-L949">createToolResultMessage() · 930–949</a></div>
      <div class="flow-return"><strong>↶ B로 돌아가 다음 모델 요청</strong><small>일반적인 도구 루프입니다. 종료 결정·중단 조건이 있으면 멈춥니다.</small><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L281-L299">finishTurn과 다음 턴 판단 · 281–299</a></div>
    </div>
    <div class="flow-track">
      <div class="flow-edge">C → 아니오 · 답변 경로 ↓</div>
      <div class="flow-node flow-node--pi"><span class="flow-owner">Pi · G</span><strong>턴을 마치고 후속 작업 확인</strong><small>대기 중인 요청이나 명시적 계속 결정이 있으면 ↶ B로 돌아갑니다.</small><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L281-L325">runLoop()의 종료 판단 · 281–325</a></div>
      <div class="flow-arrow" aria-hidden="true">↓ 자동 진행까지 끝나면</div>
      <div class="flow-node flow-node--app"><span class="flow-owner">내 프로젝트 · H</span><strong>결과를 화면에 표시 · 실행 상태 정리</strong><small>예시 <code>subscribe()</code>로 진행 상태 갱신<br>완료·실패·중단을 구분해 보여 주기</small><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/agent-session.ts#L1373-L1383">SDK 이벤트 연결: AgentSession.subscribe()</a><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/docs/cli-integration.md#control-pi-with-rpc">RPC의 전체 완료 신호: agent_settled</a></div>
      <div class="flow-exit"><span class="flow-owner">C의 별도 출구 · 모델 오류 / 중단</span><strong>현재 코어 실행 종료 → H에서 상태 표시</strong><small>도구 오류와 달리 다음 모델 요청으로 바로 돌아가지 않습니다. 세션의 자동 복구 여부도 확인합니다.</small><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L245-L255">error / aborted의 return · 245–255</a></div>
    </div>
  </div>
  <p class="note-caption">단일 요청의 일반적인 경로를 단순화했습니다. D의 검증 거절은 E를 건너뜁니다. 출력 길이가 잘려 불완전한 도구 인자가 온 경우에도 실행을 막습니다. 조향·대기열·재시도·압축의 세부 흐름은 생략했습니다. 살구색은 제안하는 앱 구조이며 Pi의 자동 제공 기능이 아닙니다.</p>
</figure>

중요한 경계는 E입니다. **Pi가 등록한 도구 호출을 검사해도, 그 뒤에서 실행한 Claude Code·Codex의 내부 도구 호출까지 Pi의 훅으로 자동 검사하는 것은 아닙니다.** CLI 안의 실행은 해당 도구의 하네스·권한 설정이 맡습니다. 앱은 두 실행 환경을 의도한 범위로 연결해야 합니다.

## 내 Claude·Codex 작업에는 어디부터 적용할까

처음에는 **검색 태그 필터 같은 기능 하나**와 완료 기준을 정하겠습니다. 구현을 맡긴 Codex에게 목표·변경 범위를 전달하고, Claude에게는 구현 결과와 같은 완료 기준을 전달합니다. 리뷰 지적이 나오면 보완 횟수를 한 번으로 제한하고 다시 확인합니다.

| 단계 | 이 프로젝트에서 만들 것 | 확인할 결과 |
| --- | --- | --- |
| 기능 범위 고정 | `Base.astro`의 검색 UI·필터, 기존 `search.json` 태그 재사용 | 새 백엔드 없이 끝낼 작은 변경인지 봅니다. |
| Codex 구현 연결 | 분리한 작업 공간에서 `runCodex()` 실행 | 구현 결과 v1과 변경 파일을 기록합니다. |
| Claude 리뷰 연결 | v1과 완료 기준을 `runClaudeReview()`에 전달 | 태그 유지 문제의 위치·이유·확인할 동작을 받습니다. |
| 보완·검사 연결 | 리뷰를 Codex에 전달하고 v2를 재검토·검사 | 최신 버전의 리뷰와 검사 결과가 모두 있는지 봅니다. |

위 순서가 늘 고정되어 있다면 작은 스크립트로도 만들 수 있습니다. Pi를 더할 이유는 **리뷰 결과에 따라 보완하거나, 추가 자료를 읽거나, 실패를 설명하는 다음 행동을 모델이 선택해야 할 때** 생깁니다. 이때도 구현자·리뷰어를 여러 개씩 늘리기보다, 하나의 작업이 제한 안에서 끝나는지 먼저 보겠습니다.

<figure class="pi-note dev-flow" aria-label="프로젝트의 작업 성격에 따라 Pi 도입과 기능 확장을 판단하는 흐름도">
  <figcaption><span class="note-index">그림 06 · 작은 도입 순서</span><strong>프로젝트의 작업 하나에서 시작해요.</strong></figcaption>
  <div class="flow-legend"><span class="flow-tag flow-tag--pi">Pi 제공 기능 선택</span><span class="flow-tag flow-tag--app">내 프로젝트의 설계·구현</span></div>
  <div class="flow-node flow-node--app"><span class="flow-owner">내 프로젝트 · 판단 1</span><strong>상황에 따라 모델이 자료·도구를 골라야 하나요?</strong><small>예: 리뷰 지적에 따른 보완·추가 자료 선택</small></div>
  <div class="flow-split"><span>↓ 예 · 아래로</span><div class="flow-node flow-node--app"><span class="flow-owner">아니오 → 여기서 마침</span><strong>고정된 스크립트·API로 먼저 해결</strong><small>예: 끊어진 링크 검사, 빌드 실행</small></div></div>
  <div class="flow-node flow-node--pi"><span class="flow-owner">Pi · 실행 기반 선택</span><strong>세션 하나 + 구현·리뷰·검사 입구 연결</strong><small>같은 TypeScript 앱은 SDK, 별도 프로세스 제어는 RPC 검토</small><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/sdk.ts#L411-L480">SDK 조립: Agent → AgentSession · 411–480</a><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/docs/cli-integration.md#control-pi-with-rpc">별도 프로세스의 RPC 연결</a></div>
  <div class="flow-arrow" aria-hidden="true">↓</div>
  <div class="flow-node flow-node--app"><span class="flow-owner">내 프로젝트 · 구현과 확인</span><strong>작업 명세·CLI 실행·로그·중단·한도 연결</strong><small>Codex 구현 → Claude 리뷰 → 필요한 보완 → 재검토<br>기준 누락·CLI 실패·중단·한도 도달도 확인</small><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L821-L839">도구 실행에 전달되는 signal · 821–839</a></div>
  <div class="flow-arrow" aria-hidden="true">↓</div>
  <div class="flow-node flow-node--app"><span class="flow-owner">내 프로젝트 · 판단 2</span><strong>최신 구현 결과에 리뷰·검사 결과가 모두 있나요?</strong></div>
  <div class="flow-split"><span>↓ 예 · 여기서 유지</span><div class="flow-node flow-node--app"><span class="flow-owner">아니오 → 부족한 결과 확인</span><strong>필요한 보완·재검토만 허용</strong><small>예: v1의 지적 → Codex 보완 → v2 리뷰·검사<br>보완 횟수 초과·리뷰 실패는 상태를 남기고 멈춤</small><a href="https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L708-L780">Pi 연결점: 도구 검증·beforeToolCall</a></div></div>
  <div class="flow-node flow-node--app"><span class="flow-owner">내 프로젝트 · 이후 개선</span><strong>실제로 놓친 작업을 보고 기능 하나씩 추가</strong><small>인계 누락은 명세부터, 자료 부족은 검색부터 보완합니다. 병렬화는 서로 독립적인 작업에만 더합니다.</small></div>
  <p class="note-caption">도입 순서는 작성자의 제안입니다. 프로젝트 상자는 구현·판단할 일을 뜻합니다. Pi 상자의 소스 링크는 선택할 수 있는 실행 기반을 가리키며, 이 흐름 전체를 자동으로 만들어 준다는 뜻은 아닙니다.</p>
</figure>

확인할 사례도 구체적으로 정하겠습니다. 문제가 있는 수정안을 잡아내는지, 검토 기준이 없을 때 무엇을 요청하는지, CLI 실패를 ‘문제 없음’으로 잘못 표시하지 않는지, 중단 뒤 프로세스가 남지 않는지 살펴봅니다. 기존 도구에 같은 명세를 직접 전달했을 때와 비교해, 추가한 연결이 인계 누락을 줄이는지도 확인할 수 있겠습니다. 이는 앞으로 검증할 항목이며 측정 결과가 아닙니다.

## Pi가 말하는 최소 구성은 어디에서 보일까

Pi의 README는 최소 구성의 확장형 하네스를 지향하며 서브에이전트·계획 모드 등을 기본 제공에서 제외한다고 설명합니다. 필요한 작업 방식은 확장할 수 있습니다. 이것은 프로젝트가 명시한 방향입니다. [README의 설명](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/README.md#L15-L21)

제가 여기서 읽은 ‘적당함’은 **공통 실행은 재사용하고 프로젝트마다 달라지는 정책은 밖에서 정하도록 둔 구성**입니다. 저장소 전체의 코드가 적다는 뜻은 아닙니다. 코어의 [`AgentOptions`](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent.ts#L114-L137)는 모델 스트림·메시지 변환·도구를 받고, [`createAgentSession()`](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/sdk.ts#L191-L209)은 SDK 구성을 조립합니다. 검토 CLI의 세부 실행을 코어 루프에 넣지 않아도 되는 경계입니다.

기본 코딩 도구 이름은 `read`, `bash`, `edit`, `write`이지만, 모든 프로젝트가 이 네 개를 그대로 써야 하는 것은 아닙니다. 이 글의 예제는 `implement_feature`·`review_change`·`run_checks`라는 업무 단위의 입구를 골랐습니다. **목표를 끝낼 수 있는 도구를 선택하되, 실행할 범위를 분명히 하자**는 적용 기준으로 읽었습니다. [기본 도구 상수](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/settings-manager.ts#L85)

## 내 하네스는 어디까지 만들면 적당할까

아래는 이 사례에 적용하는 **작성자의 구성 제안**입니다. Pi의 공식 최소 구성 목록은 아닙니다.

### 한 요청을 끝내는 경로부터 만듭니다

판단 모델 하나, Pi 세션 하나, 기능 하나로 시작하겠습니다. 구현·리뷰·검사는 작은 도구 계약으로 나누고, 우선 순차 실행과 한 번의 보완까지만 연결합니다. 병렬 실행이나 여러 리뷰어는 실제 필요가 생긴 뒤 더합니다. Pi를 통해 모델만 바꾸는 첫 번째 구조라면 CLI 연결 자체가 필요하지 않다는 점도 기억해 주세요. [모델 선택 SDK 예제](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/examples/sdk/02-custom-model.ts)

### 기록의 기준을 한 곳에 둡니다

Pi의 대화 기록은 `SessionManager`를 기준으로 보고, 앱에는 작업 ID·변경 내용의 식별자·외부 CLI의 세션 식별자를 대응시켜 남기겠습니다. 화면에 표시한 기록과 다음 모델 요청의 기록을 각각 임의로 수정하면 무엇이 전달됐는지 추적하기 어려워집니다. 외부 도구로 넘긴 작업 명세도 보관하면 인계 내용을 다시 확인할 수 있습니다. [Pi 세션 저장의 기준](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/docs/sdk.md#session-storage)

### 중단과 관찰은 첫 구성에 포함합니다

도구 이름·호출 ID·작업 ID·완료 상태·처리 시간을 기록하고, 검토 실행 횟수와 전체 실행 시간·사용량의 한도를 정하겠습니다. 이 한도는 프로젝트의 설계 제안입니다. Pi가 기본으로 모두 보장한다는 뜻은 아닙니다. `finishTurn` 같은 턴 경계의 종료 판단과 도구의 중단 신호를 실제 실행 정책에 연결할 수 있습니다. [턴 종료 계약](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/types.ts#L162-L181), [도구에 전달되는 signal](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts#L821-L839)

Pi는 파일·프로세스·네트워크·자격 증명 접근을 제한하는 내장 권한 시스템을 제공하지 않는다고 명시합니다. 도구 스키마나 `beforeToolCall`만으로 자식 CLI의 실행 환경까지 격리되는 것은 아닙니다. 이 사례에서는 작업 공간과 각 CLI의 권한을 연결 코드에서 별도로 구성해야 합니다. [Pi 권한 설명](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/README.md#permissions--containerization)

### 복잡한 기능은 관찰된 문제에 맞춰 더합니다

| 더할 기능 | 필요가 생기는 상황 | 먼저 확인할 작은 해결책 |
| --- | --- | --- |
| 리뷰·검사 병렬 실행 | 같은 버전의 독립 작업을 함께 실행해야 합니다. | 순차 실행에서 버전·결과 대응이 맞는지 먼저 봅니다. |
| 계획·서브에이전트 | 선후 관계를 반복해서 놓치거나 별도 문맥이 필요합니다. | 작업 명세와 도구 계약부터 명확히 합니다. |
| 자동 재시도 | CLI의 일시적 실패를 복구해야 합니다. | 다시 실행해도 되는 작업인지, 실행 한도가 있는지 봅니다. |
| 검색·장기 기억 | 현재 작업 밖의 많은 기록이 필요합니다. | 명시적인 명세와 현재 세션부터 사용합니다. |
| 여러 차례 보완·재검토 | 한 번의 보완으로 부족한 작업이 반복됩니다. | 한 번의 보완 흐름부터 확인하고 종료 조건·비용 한도를 정합니다. |

이미 하네스가 있는 도구를 Pi에서 다시 부르면 바깥 판단과 안쪽 실행이 추가됩니다. 실행 횟수·문맥 전달·오류 처리도 함께 늘어납니다. **정해진 순서로 한 번 전달하면 끝나는 작업에는 작은 스크립트가 충분할 수 있습니다.** Pi는 상황에 따라 자료와 도구를 선택하는 흐름을 공통으로 관리할 필요가 있을 때 검토하겠습니다.

## 이번 글에서 검증한 범위

Pi 소스와 공식 문서에서 세션 구성·문맥 변환·도구 검증·실행·결과 기록·병렬 순서를 확인했습니다. Claude Code·Codex의 비대화형 연결 입구도 공식 문서로 확인했습니다. 그림과 애니메이션은 검색 태그 필터 개발을 소재로 직접 만든 설명용 모형입니다. 작은 검색 화면은 준비한 예시 데이터를 걸러 보여 줍니다. 블로그의 실제 검색 기능에 태그 선택기를 추가한 결과는 아닙니다.

실제 Claude·OpenAI 모델 호출, Claude Code·Codex 프로세스 실행, 제품 간 대화 가져오기, 권한 격리, 비용·속도·검토 정확도 비교는 검증하지 않았습니다. `implement_feature`·`review_change`·`run_checks`와 CLI 실행 함수는 제안한 연결 구조입니다. 이 글에서 이미 동작하는 통합 기능으로 표현하지 않았습니다.

[기술 글 작성 원칙](/wiki/writing-principles/)에도 적용 사례의 역할 구분과 소스 연결 기준을 남겼습니다. 아래에는 같은 근거를 다시 찾아볼 수 있도록 출처를 모았습니다.

## 출처

Pi 링크는 2026-10-17에 확인한 동일 커밋입니다. 외부 CLI 공식 문서는 2026-10-18에 확인했으며 이후 변경될 수 있습니다.

1. [Pi 원본 저장소](https://github.com/earendil-works/pi) · [분석 커밋](https://github.com/earendil-works/pi/commit/8efd314fa3d3232ce197d8d7a6a5660530b9d83e) · [README](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/README.md)
2. [sdk.ts: 세션 조립](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/sdk.ts) · [agent-session.ts: 세션 실행·문맥](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/agent-session.ts)
3. [agent.ts: 에이전트 구성](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent.ts) · [agent-loop.ts: 검증·실행·반복](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/agent-loop.ts) · [types.ts: 종료 계약](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/agent/src/types.ts)
4. [session-manager.ts: 세션 기록](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/session-manager.ts) · [messages.ts: 모델용 메시지 변환](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/messages.ts) · [runner.ts: context 확장](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/extensions/runner.ts)
5. [SDK 공식 문서](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/docs/sdk.md) · [모델 선택 예제](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/examples/sdk/02-custom-model.ts) · [전체 구성 예제](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/examples/sdk/12-full-control.ts)
6. [Pi CLI·RPC 통합](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/docs/cli-integration.md) · [도구 정의·defineTool](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/extensions/types.ts)
7. [모델 공급자·인증 문서](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/docs/providers.md) · [Anthropic 공급자](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/ai/src/providers/anthropic.ts) · [OpenAI Codex 공급자](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/ai/src/providers/openai-codex.ts) · [ModelRuntime](https://github.com/earendil-works/pi/blob/8efd314fa3d3232ce197d8d7a6a5660530b9d83e/packages/coding-agent/src/core/model-runtime.ts)
8. [Claude Code: 비대화형 실행·출력](https://code.claude.com/docs/en/headless) · [CLI 옵션](https://code.claude.com/docs/en/cli-reference)
9. [Codex: 비대화형 실행·JSONL·권한](https://learn.chatgpt.com/docs/non-interactive-mode) · [앱에 통합하는 App Server](https://learn.chatgpt.com/docs/app-server)
