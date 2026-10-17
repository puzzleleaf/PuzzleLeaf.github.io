# puzzleleaf의 기술블로그

기존 Jekyll 블로그를 새로 구성한 Astro 7 정적 블로그입니다. 기존 글은 이전하지 않습니다. 운영 주소는 **https://blog.puzzleleaf.dev/**입니다.

## 실행

Node.js **24 LTS** 권장. 최소 버전은 22.22.0입니다. `.node-version`에 24를 지정했습니다.

```sh
npm ci
npm run dev
```

터미널에 출력되는 Local 주소를 엽니다. 기본 포트가 사용 중이면 다른 포트가 선택됩니다.

```sh
npm run check
npm run build
npm run preview
```

## 글 쓰기

```sh
npm run new-post -- my-first-note
```

`src/content/posts/my-first-note.md`에 초안이 생깁니다. 제목, 설명, 카테고리와 태그를 채운 뒤 `draft: false`로 바꾸면 공개됩니다. 미래 날짜와 초안은 빌드 결과에서 제외됩니다. 미래 날짜 글을 예약했다면 그 날짜 이후 다시 배포해야 합니다.

```yaml
---
title: "문제를 구체적으로 설명하는 제목"
description: "이 글에서 다루는 문제와 확인한 결론"
date: 2026-10-17
updated: 2026-10-17 # 실제 내용이 바뀐 경우에만 설정
category: 개발 노트
tags: [Astro, 기록]
draft: false
featured: false
sample: false
---
```

작성자와 블로그 설명은 `src/config.ts`, 도메인은 `astro.config.mjs`와 `public/CNAME`에서 관리합니다. 커스텀 도메인이 없는 프로젝트 Pages에 배포할 때는 `SITE_URL`과 `BASE_PATH` 환경 변수를 지정하세요. 글의 파일명이 URL이 되므로 공개 후에는 안정적으로 유지하세요.

소개 글 1개와 샘플 2개가 들어 있습니다. 샘플은 화면에 명시하고 `noindex`를 부여하며 사이트맵·RSS·llms.txt에서 제외합니다. 공개 글로 전환하려면 본인의 실제 내용으로 교체하고 `sample: false`로 바꾸세요. 초안으로 숨기려면 `draft: true`로 설정합니다.

## 디자인과 기능

- 반응형 글 목록, 카테고리 필터, 실제 태그별 정적 페이지
- 본문까지 검색하는 클라이언트 검색과 `Ctrl/Cmd + K`
- 시스템 설정을 따르는 초기 테마와 선택을 저장하는 다크 모드
- 글의 자동 목차, 코드 하이라이팅, 읽기 시간, 링크 복사
- 작성자 소개, 관련 글, RSS, Markdown 원문
- JavaScript를 끈 상태에서도 글 목록과 본문을 읽을 수 있는 정적 HTML

## 검색 및 AI 검색 설계

2026-10-17 기준 공식 문서를 바탕으로 구성했습니다. 시장점유율을 단정하기보다, 콘텐츠 중심의 정적 생성과 Markdown 관리라는 요구에 맞춰 최신 Astro 7.4를 선택했습니다.

| 항목 | 구현 |
| --- | --- |
| 본문 접근성 | 빌드 시 전체 본문·제목·목차·링크를 HTML로 출력 |
| 페이지 식별 | 고유 title/description, 커스텀 도메인 canonical, 한국어 lang |
| 작성자·시간 | 실제 표시와 일치하는 작성자, 게시일, 선택적 수정일 |
| 구조화 데이터 | WebSite, ProfilePage/Person, BlogPosting, BreadcrumbList |
| 발견 경로 | 실제 내부 링크, 태그 페이지, sitemap.xml, RSS |
| 크롤러 | robots.txt에서 일반 검색 및 OAI-SearchBot 허용 |
| 공유 | Open Graph, Twitter 카드, 1200×630 PNG 이미지 |
| AI 보조 탐색 | llms.txt와 글별 Markdown 원문. 노출 필수 조건으로 취급하지 않음 |
| 샘플 제외 | 샘플 글·샘플 전용 태그 noindex, 사이트맵·피드 제외 |

`OAI-SearchBot`은 검색용이며 학습용 `GPTBot`과는 별도입니다. 현재 robots.txt는 일반 크롤러를 허용하며 학습 차단 정책은 별도로 설정하지 않았습니다.

Google은 AI 검색을 위해 별도의 AI 파일이나 전용 스키마를 요구하지 않습니다. `llms.txt` 자체가 검색 순위를 높이거나 인용을 보장한다는 주장은 하지 않습니다. 구조화 데이터는 표시된 사실을 설명하며, 보이지 않는 FAQ나 가짜 경력·성과를 추가하지 않습니다.

AI 검색에서 발견될 가능성을 높이려면 실제 글에 재현 환경, 직접 검증한 결과, 원본 코드와 출처를 담고, 내용이 바뀔 때 수정일도 갱신하세요. 기술 설정만으로 색인·순위·인용을 보장할 수 없습니다.

### 배포 후 검색 등록

1. 기존 Google Search Console 소유권을 확인하고 `/sitemap.xml`을 제출합니다.
2. 첫 공개 글의 URL 검사에서 실시간 HTML과 색인 가능 여부를 확인하고 색인을 요청합니다.
3. Bing Webmaster Tools에 같은 사이트맵을 제출하고 크롤링 상태와 AI Performance의 인용 지표를 확인합니다.
4. Naver Search Advisor를 쓰고 있다면 기존 인증 파일이 유지되었는지 확인하고 사이트맵과 RSS를 갱신합니다.
5. 공개된 `robots.txt`, canonical, 글의 구조화 데이터와 OG 이미지를 다시 확인합니다.

사이트 소유자 계정 접근이 필요한 검색 등록·제출은 코드 생성만으로 완료되지 않습니다.

### 공식 참고 자료

- [Astro 7.4](https://astro.build/blog/astro-740/)
- [Astro: GitHub Pages 배포](https://docs.astro.build/en/guides/deploy/github/)
- [Google: AI 기능과 웹사이트](https://developers.google.com/search/docs/appearance/ai-features)
- [Google: 생성형 AI 검색 최적화 안내](https://developers.google.com/search/docs/fundamentals/ai-optimization-guide)
- [OpenAI: 크롤러와 검색용 OAI-SearchBot](https://developers.openai.com/api/docs/bots)
- [Bing: AI Performance](https://blogs.bing.com/webmaster/2026/2/Introducing-AI-Performance-in-Bing-Webmaster-Tools-Public-Preview/)

## GitHub Pages 배포 전환

기존 저장소 기본 브랜치는 `master`, Pages는 기존 브랜치 기반 Jekyll 빌드입니다.

1. 개편 PR을 검토하고 `master`에 병합합니다. 구형 글·테마는 이 변경에서 제거되며 Git 기록에는 남습니다.
2. 저장소 **Settings → Pages → Build and deployment → Source**를 **GitHub Actions**로 변경합니다.
3. **Actions → Deploy blog to GitHub Pages → Run workflow**로 첫 배포를 실행합니다. 이후 `master` 푸시가 자동 배포됩니다.
4. 기존 커스텀 도메인 `blog.puzzleleaf.dev`와 HTTPS 설정을 확인합니다. 도메인을 그대로 사용하므로 DNS 변경은 필요하지 않습니다.

PR 검사에서는 타입 검사와 정적 빌드를 실행합니다. `deploy.yml`은 `master` 푸시 또는 수동 실행에서만 배포합니다. `public/CNAME`과 기존 Naver 인증 파일을 유지합니다. 저장소의 기존 LICENSE도 보존합니다.

기존 글을 삭제하는 요청에 따라 이전 글 URL은 새 글로 임의 리디렉션하지 않고 404가 됩니다. 삭제된 URL의 검색 결과는 검색 서비스의 재크롤링 후 반영되며, 즉시 제거되는 것은 아닙니다.

## 공유 이미지 다시 만들기

`scripts/generate-og.mjs`의 문구를 바꾼 뒤 `node scripts/generate-og.mjs`로 `public/og.png`를 생성합니다.
