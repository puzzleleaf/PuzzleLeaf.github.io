export const site = {
  title: 'puzzleleaf의 기술블로그',
  author: 'puzzleleaf',
  description: '개발 지식, 구현 과정, 문제 해결 방법을 기록하는 puzzleleaf의 기술블로그.',
  github: 'https://github.com/puzzleleaf',
};

export const href = (path = '') => `${import.meta.env.BASE_URL.replace(/\/$/, '')}/${path.replace(/^\//, '')}`;
export const formatDate = (date: Date) => new Intl.DateTimeFormat('ko-KR', {
  year: 'numeric', month: '2-digit', day: '2-digit', timeZone: 'Asia/Seoul',
}).format(date).replace(/\.$/, '');
