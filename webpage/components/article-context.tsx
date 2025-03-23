import { Card, CardHeader, CardTitle, CardContent } from './ui/card';

interface ArticleContextProps {
  context: string;
  keyTerms: Array<{
    term: string;
    definition: string;
  }>;
}

export default function ArticleContext({ context, keyTerms }: ArticleContextProps) {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Article Context</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">{context}</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Key Terms</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="space-y-4">
            {keyTerms.map((item, index) => (
              <div key={index}>
                <dt className="font-medium text-gray-900 dark:text-gray-100">{item.term}</dt>
                <dd className="mt-1 text-gray-700 dark:text-gray-300">{item.definition}</dd>
              </div>
            ))}
          </dl>
        </CardContent>
      </Card>
    </div>
  );
}
