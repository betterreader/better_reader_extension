import { Card, CardHeader, CardTitle, CardContent } from './ui/card';

interface ArticleSummaryProps {
  summary: string;
}

export default function ArticleSummary({ summary }: ArticleSummaryProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Article Summary</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-gray-700 dark:text-gray-300 leading-relaxed">{summary}</p>
      </CardContent>
    </Card>
  );
}
