import re
from enum import Enum

class ComplexityLevel(Enum):
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3
    VERY_COMPLEX = 4

class SQLComplexityEvaluator:
    """Evaluates SQL query complexity based on various factors."""
    
    def __init__(self):
        self.complexity_score = 0
    
    def evaluate(self, query: str) -> dict:
        """
        Evaluate SQL query complexity.
        
        Args:
            query: SQL query string
            
        Returns:
            Dictionary with complexity level and score
        """
        self.complexity_score = 0
        query_upper = query.upper()
        
        self.count_joins(query_upper)
        self.count_subqueries(query_upper)
        self.count_aggregations(query_upper)
        self.check_window_functions(query_upper)
        self.check_cte(query_upper)
        self.count_conditions(query_upper)
        self.check_unions(query_upper)
        self.check_date_functions(query_upper)
        self.check_string_functions(query_upper)
        self.check_case_statements(query_upper)
        self.check_arithmetic_operations(query_upper)
        
        level = self.get_complexity_level()
        
        return {
            "score": self.complexity_score,
            "level": level.name,
            "description": self.get_description(level)
        }
    
    def count_joins(self, query: str) -> None:
        joins = len(re.findall(r'\bJOIN\b', query))
        self.complexity_score += joins * 2
    
    def count_subqueries(self, query: str) -> None:
        subqueries = len(re.findall(r'\(SELECT', query))
        self.complexity_score += subqueries * 3
    
    def count_aggregations(self, query: str) -> None:
        agg_functions = len(re.findall(r'\b(SUM|COUNT|AVG|MIN|MAX|GROUP_CONCAT)\s*\(', query))
        self.complexity_score += agg_functions * 1
        if 'GROUP BY' in query:
            self.complexity_score += 2
    
    def check_window_functions(self, query: str) -> None:
        if 'OVER' in query:
            self.complexity_score += 4
    
    def check_cte(self, query: str) -> None:
        if 'WITH' in query and 'AS' in query:
            ctes = len(re.findall(r'WITH|,\s*\w+\s+AS', query))
            self.complexity_score += ctes * 2
    
    def count_conditions(self, query: str) -> None:
        conditions = len(re.findall(r'\b(AND|OR)\b', query))
        self.complexity_score += conditions
    
    def check_unions(self, query: str) -> None:
        unions = len(re.findall(r'\bUNION', query))
        self.complexity_score += unions * 2

    def check_date_functions(self, query: str) -> None:
        date_functions = len(re.findall(r'\b(DATE|YEAR|MONTH|DAY|DATEDIFF|DATEADD|STRPTIME)\s*\(', query))
        self.complexity_score += date_functions * 1

    def check_string_functions(self, query: str) -> None:
        string_functions = len(re.findall(r'\b(CONCAT|SUBSTRING|LENGTH|TRIM|UPPER|LOWER)\s*\(', query))
        self.complexity_score += string_functions * 1

    def check_case_statements(self, query: str) -> None:
        case_statements = len(re.findall(r'\bCASE\b', query))
        self.complexity_score += case_statements * 3
    
    def check_arithmetic_operations(self, query: str) -> None:
        arithmetic_ops = len(re.findall(r'[\+\-\*/]', query))
        self.complexity_score += arithmetic_ops * 1
    
    def get_complexity_level(self) -> ComplexityLevel:
        if self.complexity_score <= 5:
            return ComplexityLevel.SIMPLE
        elif self.complexity_score <= 10:
            return ComplexityLevel.MODERATE
        elif self.complexity_score <= 15:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.VERY_COMPLEX
    
    def get_description(self, level: ComplexityLevel) -> str:
        descriptions = {
            ComplexityLevel.SIMPLE: "Basic query with minimal operations",
            ComplexityLevel.MODERATE: "Query with moderate complexity",
            ComplexityLevel.COMPLEX: "Complex query with multiple operations",
            ComplexityLevel.VERY_COMPLEX: "Very complex query requiring optimization review"
        }
        return descriptions[level]

    def rate_query(self, query: str) -> dict:
        """Quick function to rate a SQL query."""
        return self.evaluate(query)

# Example usage:
if __name__ == "__main__":
    sample_query = """
SELECT COUNT(*) AS count FROM procedure_occurrence 
WHERE DATEDIFF('day', 
    STRPTIME(procedure_datetime, '%d/%m/%Y %H:%M'), 
    STRPTIME(procedure_end_datetime, '%d/%m/%Y %H:%M')
) > 3 
AND YEAR(STRPTIME(procedure_datetime, '%d/%m/%Y %H:%M')) = 2020 
AND YEAR(STRPTIME(procedure_end_datetime, '%d/%m/%Y %H:%M')) = 2020;    """
    evaluator = SQLComplexityEvaluator()
    result = evaluator.rate_query(sample_query)
    print(result)