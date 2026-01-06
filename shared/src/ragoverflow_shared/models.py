from enum import Enum
from typing import Optional, Tuple, Dict, Any, List
from sqlmodel import SQLModel, Field as SqlField
from pydantic import BaseModel, field_validator, Field as PydanticField


class StackOverflowTag(SQLModel, table=True):
    """
    SQlModel data model for a StackOverflow tag (hashtag)
    Use `table=true` so that when we call `SQLModel.metadata.create_all(...)`, the sqlmodel library
    will automatically generate and run SQL DDL that will set up this table.
    NOTE:: We don't store the `tagId` of each tag, since the `tag_name` itself can be used as a unique identifier
    on the StackOverflow Web API: https://stackoverflow.com/questions/tagged/{`tag_name`}
    """
    tag_name: str = SqlField(primary_key=True)  # the name of the hashtag (ex. "python"), used as the PK in our DB
    count: int  # the number of posts that include this tag

    __tablename__ = "stackoverflow_tags"  # the name assigned to the tags SQL table upon creation by the ORM engine


class StackOverflowPost(SQLModel, table=True):
    """
    SQlModel data model for a StackOverflow post
    """
    # unique identifier of each post. since the primary key is an integer, sqlAlchemy wants us to
    # explicitly specify that auto-increment behavior should not be used
    post_id: int = SqlField(primary_key=True, sa_column_kwargs={"autoincrement": False})
    post_type: int = SqlField(index=True)  # corresponds to a PostType ordinal. options include questions, answers, etc
    title: str  # the title of the post
    body: str  # the content of the post
    # the tags associated w/ this post
    # format: a delimited string of tags separated by `|`, ex. `"|python|django|postgresql|...|"`
    # use the empty string "" if there are no tags
    tags: str = SqlField(index=True)
    net_votes: int = SqlField(index=True)  # computed as `amount of upvotes` minus `amount of downvotes`
    # the post ID of this post's parent. Will be null for questions, non-null for answers
    parent_id: Optional[int] = SqlField(
        default=None,
        index=True
    )

    __tablename__ = "stackoverflow_posts"  # the name assigned to the tags SQL table upon creation by the ORM engine


class PineconeVector(BaseModel):
    """
    Pydantic data model for a vector to be uploaded to PineconeDB via the pinecone library

    Pinecone expects vectors in this format:
    - id: unique string identifier
    - values: list of floats (the embedding)
    - metadata: dict of filterable attributes
    """
    id: str = PydanticField(description="Unique vector ID (post_id as string)")
    values: List[float] = PydanticField(description="384-dimensional embedding from BGE model")
    metadata: VectorMetadata = PydanticField(description="Filterable metadata stored with vector")

    @field_validator('values')
    @classmethod
    def validate_embedding_dimension(cls, v: List[float]) -> List[float]:
        """Ensure embedding has correct dimensionality (384 for BGE-small)"""
        if len(v) != 384:
            raise ValueError(f"Embedding must be 384-dimensional, got {len(v)}")
        return v

    @classmethod
    def from_dataframe_row(cls, row) -> PineconeVector:
        """
        Factory method to create a PineconeVector from a DataFrame row.

        :param row: pandas DataFrame row with post_id, embedding, post_type, net_votes, tags
        :return: PineconeVector ready for upload
        """
        return cls(
            id=str(row['post_id']),
            values=row['embedding'],
            metadata=VectorMetadata(
                post_type=int(row['post_type']),
                net_votes=int(row['net_votes']),
                tags=row['tags'] if row['tags'] else ''
            )
        )

    def to_pinecone_format(self) -> Tuple[str, List[float], Dict[str, Any]]:
        """
        Convert to Pinecone's expected tuple format: (id, values, metadata_dict)
        """
        return self.id, self.values, self.metadata.model_dump()


class VectorMetadata(BaseModel):
    """
    Metadata stored alongside each vector in Pinecone.
    These fields can be used for filtering during vector search.
    """
    post_type: int
    net_votes: int
    tags: str = PydanticField(max_length=200)

    @field_validator('tags')
    @classmethod
    def truncate_tags(cls, v: str) -> str:
        """
        Truncate tags to 200 chars at the last complete tag boundary.

        Stack Overflow tags format: |python|django|postgresql|
        We want to avoid cutting mid-tag like |python|dja
        Instead, truncate at the last complete | before 200 chars.

        Examples:
        - "|python|django|postgresql|" (50 chars) -> no truncation
        - "|python|django|postgresql|react|vue|angular|..." (250 chars)
          -> truncate at last | before char 200
        """
        if not v or len(v) <= 200:
            return v

        # Find the last pipe delimiter before position 200
        truncated = v[:200]
        last_pipe_index = truncated.rfind('|')

        if last_pipe_index > 0:
            # Truncate at the last complete tag boundary
            # Include the | to maintain format: |python|django|
            return v[:last_pipe_index + 1]

        # Fallback: if no pipes found (malformed data), return an empty string
        return ""


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
    QUESTION = 1
    ANSWER = 2
    WIKI = 3
    TAG_WIKI_EXCERPT = 4
    TAG_WIKI = 5
    MODERATOR_NOMINATION = 6
    WIKI_PLACEHOLDER = 7
    PRIVILEGE_WIKI = 8
