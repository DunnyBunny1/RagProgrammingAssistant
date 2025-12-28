from pydantic import BaseModel, field_validator
from enum import Enum


class StackOverflowTag(BaseModel):
    """
    Represents a StackOverflow Post tag
    """

    id: int  # the unique ID of the post tag
    tag_name: str  # the name of the tag, ex: `java` for `#java`
    count: int  # the number of posts that include this tag

    # make this class frozen (immutable) so that is hashable and can be used in sets / dicts
    model_config = {"frozen": True}

    # TODO: Try uncommenting this out and see if the hash works
    def __hash__(self):
        return hash((self.id, self.tag_name, self.count))

    def __eq__(self, other):
        if isinstance(other, StackOverflowTag):
            return (
                    self.id == other.id
                    and self.tag_name == other.tag_name
                    and self.count == other.count
            )


class PostType(Enum):
    """
    The different types that a stack overflow post can be
    """

    QUESTION = "question"
    ANSWER = "answer"
    OTHER = "other"


class StackOverflowPost(BaseModel):
    # Unique identifier of each post
    id: int
    # The type of the post (question, answer, etc)
    post_type: PostType

    @field_validator("post_type", mode="before")
    def convert_to_post_type(value: int):
        """
        Convert the post type from its integer value
        (used by Stack Overflow in their XML dumps) to its
        application-specific value (Question, Answer, other)

        According to the S3 Stack Overflow Data dumps, the format is
        the following:
            PostTypeId Enum8(
                'Question' = 1,
                'Answer' = 2,
                'Wiki' = 3,
                'TagWikiExcerpt' = 4,
                'TagWiki' = 5,
                'ModeratorNomination' = 6,
                'WikiPlaceholder' = 7,
                'PrivilegeWiki' = 8
            )

        Since we are only interested in questions / answers, use our custom
        PostData Type enum
        """

        if not isinstance(value, int):
            raise ValueError("Post type must be an int")
        if value == 1:
            return PostType.QUESTION
        elif value == 2:
            return PostType.ANSWER
        else:
            return PostType.OTHER
