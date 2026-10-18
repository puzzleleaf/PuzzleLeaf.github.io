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

글의 구성과 근거 표시는 [기술 글 작성 원칙](src/content/wiki/writing-principles.md)을 따릅니다. 개요와 분석 기준으로 시작하고, 관찰·해석·제안을 구분하며, 맨 아래에 출처를 모읍니다. 첫 적용 글은 [Pi 하네스 분석](src/content/posts/pi-agent-harness.md)입니다.

Pi 글의 `interactive: [pi-harness]`는 본문의 `data-pi-lab` 장면을 활성화합니다. Codex가 검색 태그 필터를 구현하고 Claude Code가 리뷰하는 과정, 작업 명세 인계, 리뷰·검사의 응답 순서를 보여 줍니다. 직접 연결과 Pi 사용의 비교 흐름도는 같은 작업의 양쪽 단계를 동기화해 강조하고, 현재 코드 짝을 보여 줍니다. 현재 단계·담당 주체·Pi 함수·파일·줄 번호를 함께 강조하고, 직접 구현할 함수와 Pi 연결점을 구분합니다. 소스 링크에 초점을 옮기거나 누르면 해당 단계에서 멈춥니다. 화면에 들어온 장면만 자동으로 반복 재생하고, 결과를 읽을 2초 간격을 둡니다. 멈춤·처음으로·조건 변경을 지원하며, 화면 밖·백그라운드·요약 탭에서는 재생을 멈춥니다. 사용자의 멈춤 선택을 유지하고 동작 줄이기 설정에서는 수동 재생만 합니다. 실제 실행 여부는 본문의 검증 범위에 적고 장면에 반복 라벨을 붙이지 않습니다. 실제 Pi·AI·CLI·파일에는 연결하지 않습니다. 해당 표식이 있는 페이지에서만 코드를 불러오고, JavaScript를 끄면 정적 요청과 결과가 표시됩니다.

설명 그림은 Markdown 본문의 `figure.pi-note`와 `src/styles/article-notes.css`로 구성합니다. 은은한 원고지 격자와 작은 SVG 그림을 사용하며, 라벨은 선택·검색할 수 있는 일반 텍스트입니다. 모바일·다크 모드에서도 읽을 수 있고 별도 JavaScript가 필요하지 않습니다. 읽기 시간은 그림의 마크업과 링크 주소를 제외한 텍스트로 계산합니다.

```sh
npm run new-post -- my-first-note
```

`src/content/posts/my-first-note.md`에 초안이 생깁니다. 제목, 설명, 카테고리와 태그를 채운 뒤 `draft: false`로 바꾸면 공개됩니다. 미래 날짜와 초안은 빌드 결과에서 제외됩니다. 미래 날짜 글을 예약했다면 그 날짜 이후 다시 배포해야 합니다.

```yaml
---
title: "문제를 구체적으로 설명하는 제목"
description: "이 글에서 다루는 문제와 확인한 결론"
summary:
  - "글의 핵심 결론"
  - "적용할 조건과 주의할 점"
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

기본 안내·샘플 글은 제거하고 Pi 분석 글과 함께 마련한 작성 원칙만 남겼습니다. 새 글을 예시로 공개할 때는 `sample: true`로 표시하면 `noindex`를 부여하고 사이트맵·RSS·llms.txt에서 제외합니다. 초안으로 숨기려면 `draft: true`로 설정합니다.

## 디자인과 기능

- 장식과 사이드바 없이 글 중심으로 구성한 반응형 화면
- 카테고리 필터, 전체 태그 목록·태그 검색과 태그별 정적 페이지
- 각 글의 본문/요약 전환 탭. `summary`에 적은 항목을 표시하며 비어 있으면 글 설명을 사용
- 본문까지 검색하는 클라이언트 검색과 `Ctrl/Cmd + K`
- 시스템 설정을 따르는 초기 테마와 선택을 저장하는 다크 모드
- 글의 자동 목차, 코드 하이라이팅, 읽기 시간, 링크 복사
- 작성자 소개, 관련 글, RSS, Markdown 원문
- JavaScript를 끈 상태에서도 글 목록과 본문을 읽을 수 있는 정적 HTML

## 개인 지식 위키

블로그 글은 `src/content/posts/`에서 날짜순으로 관리하고, 위키는 `src/content/wiki/`에서 주제별 문서로 관리합니다. 상단의 위키 메뉴에서 카테고리별 목록과 문서 검색을 사용할 수 있습니다. 상단 통합 검색과 태그는 블로그·위키를 함께 조회하며, RSS는 블로그 글만 발행합니다.

```sh
npm run new-wiki -- my-topic
```

생성된 문서의 내용을 채운 뒤 `draft: false`로 바꾸세요. 문서를 보완하면 실제 수정 날짜를 `updated`에 기록합니다. `order`가 낮은 문서가 먼저 표시되고, 같으면 제목순으로 정렬됩니다.

`related: [다른-문서-id]`를 적으면 연결 문서 목록이 만들어지며, 연결 대상에는 역참조가 표시됩니다. 역참조는 `related`에 선언한 관계를 기준으로 계산합니다. 본문 Markdown 링크는 자동 역참조 대상이 아닙니다. ID는 위키 폴더 아래 확장자를 뺀 경로입니다. 공개된 문서만 연결 목록에 표시됩니다.

위키도 본문/요약 전환, 목차, 작성·수정 날짜, Markdown 원문을 제공합니다. 사이트맵과 llms.txt에는 위키 항목도 포함됩니다. 글과 문서의 주소는 각각 `/posts/<id>/`, `/wiki/<id>/`로 구분합니다.

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

배포 기준 브랜치는 `main`입니다. 원격 저장소의 기본 브랜치와 Pages 설정은 실제 배포 전환 시 함께 변경합니다.

1. 배포를 결정한 뒤 변경 사항을 `main`에 반영하고 원격 저장소의 기본 브랜치도 `main`으로 설정합니다. 구형 글·테마는 이 변경에서 제거되며 Git 기록에는 남습니다.
2. 저장소 **Settings → Pages → Build and deployment → Source**를 **GitHub Actions**로 변경합니다.
3. **Actions → Deploy blog to GitHub Pages → Run workflow**에서 `main`을 선택해 첫 배포를 실행합니다. 이후 `main` 푸시가 자동 배포됩니다.
4. 기존 커스텀 도메인 `blog.puzzleleaf.dev`와 HTTPS 설정을 확인합니다. 도메인을 그대로 사용하므로 DNS 변경은 필요하지 않습니다.

`main` 대상 PR 검사에서는 타입 검사와 정적 빌드를 실행합니다. `deploy.yml`은 `main` 푸시 또는 `main`에서 수동 실행할 때만 배포합니다. 다른 브랜치에서 수동 실행하면 빌드·배포를 건너뜁니다. `public/CNAME`과 기존 Naver 인증 파일을 유지합니다. 저장소의 기존 LICENSE도 보존합니다.

기존 글을 삭제하는 요청에 따라 이전 글 URL은 새 글로 임의 리디렉션하지 않고 404가 됩니다. 삭제된 URL의 검색 결과는 검색 서비스의 재크롤링 후 반영되며, 즉시 제거되는 것은 아닙니다.

## 공유 이미지 다시 만들기

로고 원본은 `src/assets/puzzleleaf-logo.png`에 보관합니다. `node scripts/generate-brand-assets.mjs`로 화면용 로고·파비콘·Apple 터치 아이콘을 생성합니다. 원본의 색과 투명도를 유지하고 크기만 조정합니다. 상단 브랜드, 글의 작성자 표시, 소개 페이지는 `BrandLogo.astro`를 함께 사용합니다.

`scripts/generate-og.mjs`의 문구를 바꾼 뒤 `node scripts/generate-og.mjs`로 로고가 포함된 `public/og.png`를 생성합니다.

## 댓글: giscus

저장소 Discussions를 활성화했고 Announcements 카테고리 ID를 `src/config.ts`에 연결했습니다. giscus 앱은 `puzzleleaf/PuzzleLeaf.github.io` 하나에만 설치했습니다. 앱에는 저장소 메타데이터 읽기와 Discussions 읽기·쓰기 권한을 허용했습니다.

1. 저장소 Settings → General → Features에서 Discussions를 활성화합니다.
2. [공식 giscus 앱](https://github.com/apps/giscus)을 설치하면서 **Only select repositories**로 `puzzleleaf/PuzzleLeaf.github.io`만 선택합니다.
3. [giscus 설정](https://giscus.app/ko)에서 해당 저장소와 Announcements 유형의 댓글 카테고리를 선택합니다.
4. `src/config.ts`의 `comments`에 실제 `category`와 `categoryId`를 입력합니다. `repoId`는 저장소 API에서 확인한 값을 넣어두었습니다.
5. 로컬에서 공개 글의 댓글 영역이 로드되는지 확인합니다. 실제 댓글 등록 검증은 본인 계정으로 진행합니다.

`categoryId`가 비어 있으면 댓글 영역 자체를 표시하지 않습니다. 샘플 글에도 댓글을 표시하지 않습니다. 댓글은 화면 가까이 스크롤했을 때만 불러오며, 한국어와 블로그의 라이트·다크 테마를 따릅니다. 외부 스크립트 로드 실패 시 다시 시도할 수 있습니다.

글 제목이나 도메인을 바꿔도 같은 댓글을 찾도록 글 ID를 `specific` 매핑 키로 사용합니다. 글 ID를 바꾸면 댓글 연결도 바뀌므로 공개 후 파일명은 유지하세요. `giscus.json`에는 운영 도메인과 로컬 미리보기만 허용했습니다. giscus는 저장소 기본 브랜치의 이 파일을 읽으므로 원격에 반영되기 전에는 이 제한 설정이 적용되지 않습니다.

댓글 작성자는 GitHub 로그인과 giscus 권한 동의가 필요합니다. 댓글은 GitHub Discussions에서 관리합니다. 댓글 iframe을 본문의 검색 노출을 위한 수단으로 사용하지 않으며, 검색과 AI 검색은 정적으로 출력한 본문과 작성자·날짜·출처 정보를 중심으로 구성합니다.
