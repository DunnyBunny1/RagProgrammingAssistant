from enum import Enum

from sqlmodel import SQLModel, Field


class StackOverflowTag(SQLModel, table=True):
    """
    Data model for a StackOverflow tag (hashtag)
    Use `table=true` so that when we call `SQLModel.metadata.create_all(...)`, the sqlmodel library
    will automatically generate and run SQL DDL that will set up this table.
    NOTE:: We don't store the `tagId` of each tag, since the `tag_name` itself can be used as a unique identifier
    on the StackOverflow Web API: https://stackoverflow.com/questions/tagged/{`tag_name`}
    """
    tag_name: str = Field(primary_key=True)  # the name of the hashtag (ex. "python"), used as the PK in our DB
    count: int  # the number of posts that include this tag

    __tablename__ = "stackoverflow_tags"  # the name assigned to the tags SQL table upon creation by the ORM engine


class StackOverflowPost(SQLModel, table=True):
    """
    Data model for a StackOverflow post
    """
    # unique identifier of each post. since the primary key is an integer, sqlAlchemy wants us to
    # explicitly specify that auto-increment behavior should
    post_id: int = Field(primary_key=True, sa_column_kwargs={"autoincrement": False})
    post_type: int = Field(index=True)  # the type of the post (question, answer, etc)
    title: str  # the title of the post
    body: str  # the content of the post
    # the tags associated w/ this post
    # format: a delimited string of tags separated by `|`, ex. `"|python|django|postgresql|...|"`
    # use the empty string "" if there are no tags
    tags: str = Field(index=True)
    net_votes: int = Field(index=True)  # computed as `amount of upvotes` minus `amount of downvotes`

    __tablename__ = "stackoverflow_posts"  # the name assigned to the tags SQL table upon creation by the ORM engine


class PostType(Enum):
    """
    The different types that a stack overflow post can be
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
    """
    QUESTION = "question"
    ANSWER = "answer"
    OTHER = "other"
