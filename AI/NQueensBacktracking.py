def SolveNQeens(n):
    def backtrack(row,hills,dales,cols,res):
        if row==n:
            for c in cols:
                res.append("."*c+"Q"+"."*(n-c-1))
            res.append("")
            return
        for col in range(n):
            if col in cols or row - col in hills or row +col in dales:
                continue
            backtrack(row+1,
                     hills+[row-col],
                     dales+[row+col],
                     cols+[col],
                     res)
    res=[]
    backtrack(0,[],[],[],res)
    return res
solveNQueens(4)