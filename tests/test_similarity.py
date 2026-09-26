from eval_harness.metrics.similarity import changed_lines, patch_similarity, touched_files

A = """diff --git a/web/x.ts b/web/x.ts
--- a/web/x.ts
+++ b/web/x.ts
@@ -1,3 +1,4 @@
-if (idx >= 1) returning += 1;
+// 1-based index
+if (idx >= 2) returning += 1;
"""
B = """diff --git a/web/x.ts b/web/x.ts
--- a/web/x.ts
+++ b/web/x.ts
@@ -1,3 +1,4 @@
-if (idx >= 1)   returning += 1;
+const FIRST = 2;
+if (idx >= FIRST) returning += 1;
"""
C = """diff --git a/web/y.ts b/web/y.ts
--- a/web/y.ts
+++ b/web/y.ts
@@ -1 +1 @@
-a
+b
"""


def test_normalization_drops_comments_and_collapses_whitespace() -> None:
    assert changed_lines(A) == ["-if (idx >= 1) returning += 1;", "+if (idx >= 2) returning += 1;"]
    assert touched_files(A) == {"web/x.ts"}


def test_similarity_orders_sensibly() -> None:
    assert patch_similarity(A, A) == 1.0
    assert patch_similarity(A, C) == 0.0
    mid = patch_similarity(A, B)
    assert 0.5 < mid < 1.0
    assert patch_similarity("", A) == 0.0
