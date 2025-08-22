@echo off
echo Pushing Customer Churn Pipeline to GitHub...
echo.

echo Step 1: Pushing commits...
git push -u origin master

echo.
echo Step 2: Pushing tags (for data versioning)...
git push --tags

echo.
echo Step 3: Verifying repository...
git remote -v

echo.
echo ✅ Complete! Your repository is now available at:
echo https://github.com/nishbhas874/dmml_churn_pipeline
echo.
echo Share this URL with evaluators for assignment grading.
pause
