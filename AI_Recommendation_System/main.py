
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    # Load the career skills dataset
    try:
        data = pd.read_csv("raw_skills.csv")
    except FileNotFoundError:
        print("Error: raw_skills.csv file not found!")
        return

    # Check required columns
    if not {"job_role", "skills"}.issubset(data.columns):
        print("Error: CSV must contain job_role and skills columns.")
        return

    # Clean the dataset
    data["job_role"] = data["job_role"].fillna("").astype(str)
    data["skills"] = (
        data["skills"]
        .fillna("")
        .astype(str)
        .str.replace(";", " ", regex=False)
    )

    data = data[data["job_role"].str.strip() != ""]

    if data.empty:
        print("No career roles found in the dataset.")
        return

    print("\n================================")
    print("   AI TECH STACK RECOMMENDER")
    print("================================")
    print("Enter at least 3 different skills.")
    print("Example: Python, SQL, Statistics")

    user_input = input("\nEnter your skills: ").strip()

    user_skills = [
        skill.strip()
        for skill in user_input.split(",")
        if skill.strip()
    ]

    unique_skills = {skill.lower() for skill in user_skills}

    if len(unique_skills) < 3:
        print("Please enter at least 3 different skills.")
        return

    # Convert career skills into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    role_vectors = vectorizer.fit_transform(data["skills"])

    # Convert user skills into a vector
    user_vector = vectorizer.transform([" ".join(user_skills)])

    # Calculate cosine similarity
    scores = cosine_similarity(
        user_vector, role_vectors
    ).flatten()

    data["similarity_score"] = scores

    # Rank the career roles
    ranked = data.sort_values(
        by="similarity_score",
        ascending=False
    ).head(3)

    if ranked["similarity_score"].max() == 0:
        print("\nNo matching skills found in the dataset.")
        print("Try skills such as Python, Java, SQL, C++, or HTML.")
        return

    print("\n================================")
    print("  TOP 3 CAREER RECOMMENDATIONS")
    print("================================")

    recommendations = []

    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        score = row["similarity_score"] * 100

        print(
            f"\n{rank}. {row['job_role']}"
            f"\n   Similarity Score: {score:.2f}%"
            f"\n   Required Skills: {row['skills']}"
        )

        recommendations.append(row["job_role"])

    # Collect feedback for the recommendations
    print("\n================================")
    print("        USER FEEDBACK")
    print("================================")
    print("Rate each recommendation from 1 to 5.")
    print("1 = Not useful, 5 = Very useful")

    ratings = []

    for role in recommendations:
        while True:
            rating_input = input(
                f"Rate {role} (1-5): "
            ).strip()

            try:
                rating = int(rating_input)

                if 1 <= rating <= 5:
                    ratings.append((role, rating))
                    break
                else:
                    print("Please enter a number from 1 to 5.")

            except ValueError:
                print("Invalid input. Please enter a number.")

    print("\n================================")
    print("       FEEDBACK SUMMARY")
    print("================================")

    for role, rating in ratings:
        print(f"{role}: {rating}/5")

    average_rating = sum(
        rating for _, rating in ratings
    ) / len(ratings)

    print(f"\nAverage user rating: {average_rating:.2f}/5")
    print("\nThank you for using the recommender!")


if __name__ == "__main__":
    main()
