#!/usr/bin/env bash
# Lists the test cases tagged [!mayfail] that pass.
#
# A passing [!mayfail] case is one of two things: the functions it covers have been implemented and its tag should
# now be removed, so that it guards them against regressions; or it passes for the wrong reason, e.g. a negative
# check against a placeholder that returns false, and the test needs fixing. Either way it deserves a look.
#
# Catch2's XML output marks an expected failure as a successful test case, so failures are detected per assertion.
# With no tagged cases left, the filter matches nothing, which Catch2 treats as an error unless told otherwise.

set -euo pipefail

binary="${1:?usage: $0 <path to the all_tests binary>}"

"$binary" "[!mayfail]" --allow-running-no-tests --reporter xml 2>/dev/null | awk '
	/<TestCase / {
		name = $0
		sub(/.*<TestCase name="/, "", name)
		sub(/".*/, "", name)
		failed = 0
		next
	}
	/success="false"/ || /<Exception/ || /<FatalErrorCondition/ { failed = 1 }
	/<\/TestCase>/ { if(!failed) print name }
'
